import torch
import torch.nn as nn

from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from einops import rearrange


class iTFDNN(pl.LightningModule):
    def __init__(self, amp_model: type, amp_model_kwargs: dict, phase_model: type, phase_model_kwargs: dict, mean_model: type, mean_model_kwargs: dict, n_in: int, n_out: int,window_size: int, optimizer_config, hop_length=None, warm_start_mean_model=10) -> None:
        super().__init__()
        self.window_size = window_size
        self.n_to_predict = self.window_size // 2 # Number of fourier coefficients to predict (excl. those given by symmetry)
        self.n_in = n_in
        self.n_out = n_out
        self.amp_model = amp_model(n_in=n_in * window_size, n_out=(self.n_to_predict*n_out), **amp_model_kwargs)
        self.phase_model = phase_model(n_in=n_in * window_size, n_out=(self.n_to_predict*n_out), **phase_model_kwargs)
        self.mean_model = mean_model(n_in=n_in, n_out=n_out, **mean_model_kwargs)
        if hop_length is None:
            self.hop_length = window_size // 4
        else:
            self.hop_length = hop_length
        self.optimizer_config = optimizer_config
        self.warm_start_mean_model = warm_start_mean_model
        self.LOG_PLOTS_EVERY_N_EPOCHS = 500
        self.init_phase_xavier()

    def init_phase_xavier(self):
        nn.init.xavier_uniform_(self.phase_model.linear_out.weight)

    def make_indices_stft(self, end, window_size, hop_length, start=0):
        start_indices = np.arange(start, end, hop_length)[:-1] # last element can be out of bounds
        interval = np.arange(window_size)
        indices = np.add.outer(start_indices, interval)
        indices = indices[(indices < end).all(axis=1)]
        return indices

    def forward(self, u, return_offset=False, return_spectrum=False, return_phase_change=False) -> torch.Tensor:
        assert len(u.shape) == 3 # (batch, time, channel)
        u_shape = u.shape
        x_offset = self.mean_model(u)

        indices = self.make_indices_stft(u.shape[-2], self.window_size, self.hop_length)
        u = torch.stack([u[..., ind, :] for ind in indices], dim=-3) # stack between batch and time axes
        if return_phase_change:
            f, phi = self.predict_spectrum(u, return_phase_change=True)  # (b, h, f, c)
        else:
            f = self.predict_spectrum(u, return_phase_change=False)  # (b, h, f, c)

        f = rearrange(f, 'b h c f -> (b c) f h', b=u_shape[0])
        x = torch.real(torch.istft(f, n_fft=self.window_size, hop_length=self.hop_length, center=False))
        x = rearrange(x, '(b c) t -> b t c', b=u_shape[0], c=self.n_out)

        # Some data is lost during the stft-istft transformation
        # so this can be shorter than the original signal.
        x_offset = x_offset[:, :x.shape[1]]
        x = x_offset + x

        result = (x,) # (b t c)

        if return_offset:
            result = result + (x_offset,)

        if return_spectrum:
            result = result + (f, ) # (b, h, f, c)

        if return_phase_change:
            result = result + (phi,) # (b, h, f, c)
        
        if len(result) == 1:
            return result[0]
        return result

    def predict_spectrum(self, u, return_phase_change=False) -> torch.Tensor:
        u = rearrange(u, 'b h t c -> b h (t c)') #u.reshape(u.shape[:-2] + (-1,)) # (b, h, t * c)

        amp = self.amp_activation(self.amp_model(u)) # ( (b h) f c )
        phi = self.phi_activation(self.phase_model(u)) # ( (b h) f c )

        amp = rearrange(amp, 'b h (c f) -> b h c f', c=self.n_out, f=self.n_to_predict) # b h c f
        phi = rearrange(phi, 'b h (c f) -> b h c f', c=self.n_out, f=self.n_to_predict) # b h c f
        phi = phi[:, 1:] # Drop initial point

        init_phi = 2*torch.pi*torch.rand((phi.shape[0], 1) + phi.shape[2:]).to(phi.device)
        phi_sum = torch.cumsum(torch.cat((init_phi, phi), dim=1), dim=-3) # along h
        f = self.amp_phi_to_complex(amp, phi_sum) # b h c f

        if self.window_size % 2 == 0:
            sym_f = torch.conj(f[..., :-1])
        else:
            sym_f = torch.conj(f)
        sym_f = torch.flip(sym_f, dims=(-2,))

        zeroth_coeff = torch.zeros(f.shape[:-1] + (1,), device=f.device)
        f = torch.cat((zeroth_coeff, f, sym_f), dim=-1)
        if return_phase_change:
            return f, rearrange(phi, 'b h c f -> b h f c')
        return f

    def training_step(self, batch, batch_idx):
        u, y_true = batch
        assert len(u.shape) == 3 # (batch, time, channel)
        assert len(y_true.shape) == 3 # (batch, time, channel)
        start_time = torch.randint(self.hop_length, size=(1,)).item()
        u = u[:, start_time:]
        y_true = y_true[:, start_time:]
        y_shape = y_true.shape
        y, y_offset, f_pred, phi_pred_change = self(u, return_offset=True, return_spectrum=True, return_phase_change=True)

        offset_loss = 10 * torch.nn.functional.mse_loss(y_offset, y_true[:, :y_offset.shape[1]])
        
        if self.warm_start_mean_model < self.current_epoch:  
            f_pred = rearrange(f_pred, '(b c) f h -> b h f c', b=u.shape[0], c=self.n_out)
            # Some data is lost during the stft-istft transformation
            # so this can be shorter than the original signal.
            y_true = y_true[:, :y_offset.shape[1]] - y_offset # Subtract the mean model prediction

            y_c = rearrange(y_true, 'b t c -> (b c) t')
            f_true = torch.stft(y_c, n_fft=self.window_size, hop_length=self.hop_length, center=False, onesided=False, return_complex=True)
            f_true = rearrange(f_true, '(b c) f h -> b h f c', b=u.shape[0], c=self.n_out)

            amp_pred, _ = self.complex_to_amp_phi(f_pred[..., 1:(self.n_to_predict+1), :])
            amp_true, phase_true = self.complex_to_amp_phi(f_true[..., 1:(self.n_to_predict+1), :])

            amp_loss = torch.nn.functional.mse_loss(amp_pred, amp_true)

            tri_loss = lambda x1, x2: torch.mean(2*torch.sin((x1 - x2) / 2)**2)
            phi_true_change = (phase_true[:, 1:] - phase_true[:, :-1]) # inverse of cumsum

            phi_loss = 0.1*tri_loss(phi_pred_change, phi_true_change)

            loss_ = amp_loss + offset_loss + phi_loss
            self.log("loss/train", loss_.item(), on_step=False, on_epoch=True)
            self.log("loss/train/amp", amp_loss.item(), on_step=False, on_epoch=True)
            self.log("loss/train/phi", phi_loss.item(), on_step=False, on_epoch=True)
            self.log("loss/train/offset", offset_loss.item(), on_step=False, on_epoch=True)
        else:
            loss_ = offset_loss
            amp_loss = torch.Tensor([torch.nan])
            phi_loss = torch.Tensor([torch.nan])
            self.log("loss/train/offset", offset_loss.item(), on_step=False, on_epoch=True)

        
        lightning_optimizer = self.trainer.optimizers[0]
        for param_group in lightning_optimizer.param_groups:
            lr = param_group['lr']
            break
        self.log("learning_rate", lr)
        return loss_

    def validation_step(self, batch, batch_idx):
        u, y_true = batch
        assert len(u.shape) == 3 # (batch, time, channel)
        assert len(y_true.shape) == 3 # (batch, time, channel)
        _, y_offset, f_pred, phi_pred_change = self(u, return_offset=True, return_spectrum=True, return_phase_change=True)

        offset_loss = torch.nn.functional.mse_loss(y_offset, y_true[:, :y_offset.shape[1]])
        
        f_pred = rearrange(f_pred, '(b c) f h -> b h f c', b=u.shape[0], c=self.n_out)
        # Some data is lost during the stft-istft transformation
        # so this can be shorter than the original signal.
        y_true = y_true[:, :y_offset.shape[1]] - y_offset # Subtract the mean model prediction

        y_c = rearrange(y_true, 'b t c -> (b c) t')
        f_true = torch.stft(y_c, n_fft=self.window_size, hop_length=self.hop_length, center=False, onesided=False, return_complex=True)
        f_true = rearrange(f_true, '(b c) f h -> b h f c', b=u.shape[0], c=self.n_out)

        amp_pred, _ = self.complex_to_amp_phi(f_pred[..., 1:(self.n_to_predict+1), :])
        amp_true, phase_true = self.complex_to_amp_phi(f_true[..., 1:(self.n_to_predict+1), :])

        amp_loss = torch.nn.functional.mse_loss(amp_pred, amp_true)

        tri_loss = lambda x1, x2: torch.mean(2*torch.sin((x1 - x2) / 2)**2)
        phi_true_change = (phase_true[:, 1:] - phase_true[:, :-1]) # inverse of cumsum

        phi_loss = 0.1*tri_loss(phi_pred_change, phi_true_change)

        loss_ = amp_loss + offset_loss + phi_loss
        self.log("loss/validation", loss_.item(), on_step=False, on_epoch=True)
        self.log("loss/validation/amp", amp_loss.item(), on_step=False, on_epoch=True)
        self.log("loss/validation/phi", phi_loss.item(), on_step=False, on_epoch=True)
        self.log("loss/validation/offset", offset_loss.item(), on_step=False, on_epoch=True)

        if batch_idx==0 and self.current_epoch != 0 and self.current_epoch % self.LOG_PLOTS_EVERY_N_EPOCHS == 0:
            self.validation_plots()
        return loss_

    def validation_plots(self):
        if not isinstance(self.trainer.logger, pl.loggers.wandb.WandbLogger):
            print("Logging figures only possible when logging to WandB")
            return
            
        predict_dataset = self.trainer.datamodule.plot_dataloader().dataset
        plots = {}
        for idx in range(len(predict_dataset)):
            meta, data = predict_dataset.get_item_for_plot(idx)
            (title, _, target_columns) = meta
            (X_, y_) = data
            X_ = X_.unsqueeze(0)
            y_pred, y_offset = self(X_.to(self.device), return_offset=True)
            y_pred, y_offset = y_pred.cpu().squeeze(), y_offset.cpu().squeeze()

            f_true = torch.stft(rearrange(y_, 't c -> c t'), n_fft=self.window_size, hop_length=self.hop_length, center=False, onesided=False, return_complex=True)
            y_, y_pred, f_true = y_.numpy(), y_pred.detach().numpy(), f_true.numpy()
            
            for j, col in enumerate(target_columns):
                fig = plt.figure()
                ax = fig.gca()
                ax.set_title(title + ' ' + col)
                ax.plot(y_pred[..., j], label='approximation')
                ax.plot(y_[..., j], label='ground truth')
                ax.plot(y_offset[..., j], label='offset approximation')
                ax.legend(loc='upper left')
                ax.grid(True)
                plots[title + '/' + col] = fig

        self.trainer.logger.experiment.log(plots)
        
    def amp_activation(self, amp: torch.Tensor):
        return torch.nn.functional.elu(amp) + 1.0

    def phi_activation(self, phi: torch.Tensor):
        return 2 * torch.pi * 1/(1+torch.exp(-phi)) - torch.pi

    def make_amp_phi(self, x: torch.Tensor):
        amp = self.amp_activation(x[..., 0])
        phi = self.phi_activation(x[..., 1])
        return amp, phi

    def amp_phi_to_complex(self, amp: torch.Tensor, phi: torch.Tensor):
        return amp * torch.exp(1.j * phi)
    
    def complex_to_amp_phi(self, comp: torch.Tensor):
        amp = torch.abs(comp)
        phi = torch.angle(comp) % (2 * torch.pi) - torch.pi
        return amp, phi

    def configure_optimizers(self):
        optimizer = self.optimizer_config["optimizer"](
            self.parameters(),
            **self.optimizer_config["optimizer_init_kwargs"]
        )

        try:
            scheduler = self.optimizer_config["scheduler"]
            scheduler = scheduler(optimizer, **self.optimizer_config["scheduler_init_kwargs"])
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        except KeyError:
            return optimizer

class ResDNN(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, num_layers):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear_in = nn.Linear(n_in, n_hidden)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(n_hidden, n_out)
        hidden = []
        for _ in range(num_layers-2):
            hidden.append(
                    nn.Linear(n_hidden, n_hidden)
            )
        
        self.hidden = nn.ModuleList(hidden)


    def forward(self, in_: torch.Tensor):
        x = self.linear_in(in_)
        for layer in self.hidden:
            x = x + layer(x)
            x = nn.functional.relu(x)
        x = self.linear_out(x)
        return x
