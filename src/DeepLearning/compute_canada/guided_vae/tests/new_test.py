# import argparse
import os
import random
import pandas as pd
from glob import glob
import torch
import numpy as np
from scipy import stats
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
# from torchvision.io import write_video
from torchvision.utils import make_grid, save_image
import imageio

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from matplotlib.colors import ListedColormap

from ..reconstruction import AE
from ..datasets import MeshData
from ..utils import utils, DataLoader, mesh_sampling, sap
from .disentanglement_lib.sap_score import _compute_sap
from .render_outputs import Renderer
from .proportions import calculate_distances_in_folder, add_proportions_age_gender_to_csv, distance_proportion_averages

class Tester:
    def __init__(self, model_manager, norm_dict,
                 train_load, val_load, test_load, 
                 out_dir, config, logging, model, device, 
                 meshdata, dataset_metadata_path):

        self._model = model
        self._device = device
        self._out_dir = out_dir
        self._train_loader = train_load
        self._val_loader = val_load
        self._test_loader = test_load
        self._latent_stats = self.compute_latent_stats(train_load)
        self._std = meshdata.std
        self._mean = meshdata.mean
        self._template_path = meshdata.template_fp
        self._dataset_metadata_path = dataset_metadata_path
        self._renderer = Renderer(img_size=256, rend_device=device)
        self._age_to_z = self.build_age_z_lookup()
        self._train_all_latents_list, self._train_id_latents_list, self._train_gt_ages_list, self._train_re2_age_list, self._train_fname_list, self._train_dataset_list = self.process_data(train_load)
        self._val_all_latents_list, self._val_id_latents_list, self._val_gt_ages_list, self._val_re2_age_list, self._val_fname_list, self._val_dataset_list = self.process_data(val_load)
        self._test_all_latents_list, self._test_id_latents_list, self._test_gt_ages_list, self._test_re2_age_list, self._test_fname_list, self._test_dataset_list = self.process_data(test_load)

    def __call__(self):

        # eval_loader = self._test_loader # self._val_loader,
        eval_loader_name = 'TEST' # 'VAL'

        self._renderer.set_renderings_size(512)
        self._renderer.set_rendering_background_color([1, 1, 1])

        self.render_dataloader_sanity_check(self._test_loader, n_samples=16)
        self.reconstruction_sanity_check(n_samples=8)
        self.random_generation_and_rendering(n_samples=16)

        static_ages = [0, 4, 8, 12, 17]
        for age in static_ages:
            self.random_generation_and_rendering(n_samples=16, age=age)

        self.per_variable_range_experiments()
        self.age_encoder_decoder_accuracy(eval_loader_name=eval_loader_name)
        self.age_prediction_MLP(eval_loader_name=eval_loader_name)
        self.age_latent_changing(eval_loader_name=eval_loader_name)
        self.tsne_visualization()
        self.stats_tests_correlation()
        self.proportions(eval_loader_name=eval_loader_name)
        self.plot_proportions()


    def compute_latent_stats(self, data_loader):
        print("Running compute_latent_stats")
        storage_path = "/raid/compass/athena/outputs/sc-vae/z_stats.pkl"
        try:
            with open(storage_path, 'rb') as file:
                z_stats = pickle.load(file)
                print("already exists")
        except FileNotFoundError:
            latents_list = []
            for data in tqdm.tqdm(data_loader):
                mu, _ = self._model.encoder(data.x.to(self._device))
                latents_list.append(mu)
            latents = torch.cat(latents_list, dim=0)
            z_means = torch.mean(latents, dim=0)
            z_stds = torch.std(latents, dim=0)
            z_mins, _ = torch.min(latents, dim=0)
            z_maxs, _ = torch.max(latents, dim=0)
            z_stats = {'means': z_means, 'stds': z_stds,
                       'mins': z_mins, 'maxs': z_maxs}

            with open(storage_path, 'wb') as file:
                pickle.dump(z_stats, file)
        print("Finished...")
        return z_stats


    def build_age_z_lookup(self, z_dim=1, n_grid=2000, z_range_sigma=4.0):
        """
        Build a reusable lookup between a scalar target (for example age in
        years) and one latent dimension by sweeping z_dim and evaluating the
        regression head. This uses model.reg_2(z), so no decoder pass is needed.
        """

        z_means = self._latent_stats["means"].to(self._device)
        z_stds = self._latent_stats["stds"].to(self._device)

        center = float(z_means[z_dim].item())
        spread = float(z_stds[z_dim].item())
        if spread == 0:
            raise RuntimeError(f"Latent std for z[{z_dim}] is 0, cannot build lookup.")
       
        z_values = torch.linspace(
            center - z_range_sigma * spread,
            center + z_range_sigma * spread,
            n_grid,
            device=self._device,
        )

        z_grid = z_means.unsqueeze(0).repeat(n_grid, 1)
        z_grid[:, z_dim] = z_values

        with torch.no_grad():
            pred_values = self._model.reg_2(z_grid).view(-1).detach().cpu()

        # Create mapping for integers 0-17
        mapping = {}
        for target in range(18):
            distances = torch.abs(pred_values - target)
            closest_idx = torch.argmin(distances).item()
            mapping[target] = {
                "z_value": z_values[closest_idx].item(),
                "predicted_value": pred_values[closest_idx].item()
            }

        return mapping
    

    ## maybe put this in model manager to make it more general and call that in here
    def process_data(self, loader):

        all_latents_list = []
        id_latent_list = []
        gt_ages_list = []
        re2_age_list = []
        fname_list = []
        dataset_list = []

        metadata = pd.read_csv(self._dataset_metadata_path)

        for batch in tqdm.tqdm(loader):
            data = batch.x
            gt_age = batch.y[:,:,2]
            fname = batch.fname
            _, z, _, _, re_2 = self._model(data.to(self._device))
    
            for i in tqdm.tqdm(range(z.shape[0])):

                all_latents_list.append(z[i].tolist())
                id_latent_list.append(z[i][2:].tolist())
                gt_ages_list.append(gt_age[i].tolist())
                re2_age_list.append(re_2[i].tolist())

                fname_int = int(fname[i].item())
                fname_str = f"f_{fname_int}"
                fname_list.append(fname_str)
                dataset_list.append(metadata[metadata['id'] == fname_str]['Dataset'].values[0])

                # fname_list.append(fname[i].tolist())
                # dataset_list.append(metadata[metadata['id'] == fname[i]]['Dataset'].values[0])

        return all_latents_list, id_latent_list, gt_ages_list, re2_age_list, fname_list, dataset_list

    def _unnormalize_verts(self, verts):

        return verts * self._std.to(verts.device) + self._mean.to(verts.device)

    def render_dataloader_sanity_check(self, data_loader=None,
                                       n_samples: int = 16,
                                       file_name: str = "dataloader_sanity.png"):
        """
        Take a few meshes from the dataloader, unnormalize, render with
        Renderer.render, and save them as a grid. Use this to debug distortion.
        """
        if data_loader is None:
            data_loader = self._test_loader

        self._model.eval()

        samples = []
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                # batch.x assumed to be [B, N, 3] normalized verts
                verts = batch.x.to(self._device)
                verts = self._unnormalize_verts(verts)
                samples.append(verts)
                total += verts.shape[0]
                if total >= n_samples:
                    break

        if not samples:
            raise RuntimeError("No meshes were loaded from the dataloader.")

        verts = torch.cat(samples, dim=0)[:n_samples]  # [n_samples, N, 3]

        # Render -> [B, 3, H, W] on GPU
        renderings = self._renderer.render(verts)      # torch, no need to .cpu() yet

        # Make grid and save
        grid = make_grid(renderings.cpu(), nrow=4, padding=10, pad_value=1.0)
        file_path = os.path.join(self._out_dir, file_name)
        save_image(grid, file_path)
        print(f"Saved dataloader render sanity check to {file_path}")

    def reconstruction_sanity_check(self, n_samples=8):
        self._model.eval()
        xs = []
        recons = []
        with torch.no_grad():
            for batch in self._test_loader:
                x = batch.x.to(self._device)              # normalized
                # mu, _ = self._model.encoder(x)
                # out = self._model.decoder(mu)
                out, mu, log_var, _, _ = self._model(x)  
                rmse = torch.mean((x - out) ** 2).sqrt().item()
                print(f"Reconstruction RMSE (normalized space): {rmse:.4f}")
                xs.append(self._unnormalize_verts(x))
                recons.append(self._unnormalize_verts(out))
                if len(xs) * x.shape[0] >= n_samples:
                    break
        xs = torch.cat(xs, dim=0)[:n_samples]
        recons = torch.cat(recons, dim=0)[:n_samples]

        orig_r = self._renderer.render(xs).cpu()
        recon_r = self._renderer.render(recons).cpu()

        grid = make_grid(torch.cat([orig_r, recon_r], dim=0),
                        nrow=n_samples, padding=10, pad_value=1.0)
        file_path = os.path.join(self._out_dir, 'recon_sanity.png')
        save_image(grid, file_path)
        print(f"Saved reconstruction sanity check to {file_path}")

    def random_latent(self, n_samples, z_range_multiplier=1, age=None):
        z_means = self._latent_stats['means']
        z_mins = self._latent_stats['mins'] * z_range_multiplier
        z_maxs = self._latent_stats['maxs'] * z_range_multiplier

        uniform = torch.rand([n_samples, z_means.shape[0]],
                                device=z_means.device)
        z = uniform * (z_maxs - z_mins) + z_mins

        if age is not None:
            z_value_for_age = self._age_to_z[age]["z_value"]
            z[:, 1] = z_value_for_age

        return z

    def random_generation(self, n_samples=16, z_range_multiplier=1, age=None, denormalize=True):
        with torch.no_grad():
            z = self.random_latent(n_samples, z_range_multiplier, age=age)
            gen_verts = self._model.decoder(z.to(self._device))
            if denormalize:
                gen_verts = self._unnormalize_verts(gen_verts)
        return gen_verts

    def random_generation_and_rendering(self, n_samples=16, z_range_multiplier=1, age=None):
        gen_verts = self.random_generation(n_samples, z_range_multiplier, age=age)
        renderings = self._renderer.render(gen_verts).cpu()
        grid = make_grid(renderings, padding=10, pad_value=1)
        if age is None:
            file_path = os.path.join(self._out_dir, 'random_generation.png')
        else:
            file_path = os.path.join(self._out_dir, f'random_generation_age_{age}.png')
        save_image(grid, file_path)

    def per_variable_range_experiments(self):

        print("Running per_variable_range_experiments")

        latent_size = 12
        age_latent_size = 1
        disease_latent_size = 1
        feature_latent_size = latent_size - age_latent_size - disease_latent_size

        z_means = self._latent_stats['means']
        z_mins = self._latent_stats['mins']
        z_maxs = self._latent_stats['maxs']

        ##### AGE TEST #####

        # change only the age latent variables all at the same time from their own min to max

        n_steps_age = 18
        # z_age_mins = float(z_mins[1])
        # z_age_maxs = float(z_maxs[1])
        z_age_mins = float(z_maxs[1])
        z_age_maxs = float(z_mins[1])

        z = z_means.repeat(n_steps_age, 1)
        z[:, 1] = torch.linspace(z_age_mins, z_age_maxs, n_steps_age).to(self._device)

        gen_verts = self._model.decoder(z.to(self._device))
        gen_verts = self._unnormalize_verts(gen_verts)

        differences_from_first = self._renderer.compute_vertex_errors(gen_verts, gen_verts[0].expand(gen_verts.shape[0], -1, -1))
        renderings = self._renderer.render(gen_verts).detach().cpu()
        differences_renderings = self._renderer.render(gen_verts, differences_from_first, error_max_scale=5).cpu().detach()
        frames = torch.cat([renderings, differences_renderings], dim=-1)
        all_frames_age = torch.cat([frames, torch.zeros_like(frames)[:2, ::]])
            
        
        file_path_age = os.path.join(self._out_dir, 'latent_exploration_all_age_latents_[z_max-min].mp4')
        # write_video(file_path_age, all_frames_age.permute(0, 2, 3, 1) * 255, fps=4)       
        frames_np = (all_frames_age.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        imageio.mimsave(file_path_age, frames_np, fps=4)

        # changing all age latent values [0-17]

        # for each z value in self._age_to_z, change z[:, 1] for each of these
        # z = z_means.repeat(len(self._age_to_z), 1)
        for age, values in self._age_to_z.items():
            z[age, 1] = values["z_value"]

        gen_verts = self._model.decoder(z.to(self._device))
        gen_verts = self._unnormalize_verts(gen_verts)

        differences_from_first = self._renderer.compute_vertex_errors(
            gen_verts, gen_verts[0].expand(gen_verts.shape[0], -1, -1)
        )
        renderings = self._renderer.render(gen_verts).detach().cpu()
        differences_renderings = self._renderer.render(gen_verts, differences_from_first, error_max_scale=5).cpu().detach()

        frames = torch.cat([renderings, differences_renderings], dim=-1)

        file_path = os.path.join(self._out_dir, 'latent_exploration_all_age_latents_[0-17].mp4')
        # write_video(file_path, frames.permute(0, 2, 3, 1) * 255, fps=4)
        frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        imageio.mimsave(file_path, frames_np, fps=4)

        # Create a .png with all meshes in one row and difference maps in the second row
        combined_frames = torch.cat([renderings, differences_renderings], dim=0)
        age_range = range(0, 18)
        grid = make_grid(combined_frames, nrow=len(age_range), padding=10, pad_value=1)
        file_path_png = os.path.join(self._out_dir, 'latent_exploration_all_age_latents_[0-17].png')
        save_image(grid, file_path_png)

        #### NOT AGE TESTS ####
        n_steps = 10
        all_frames, all_rendered_differences, max_distances = [], [], []
        all_renderings = []
        # change each latent variable one by one
        for i in tqdm.tqdm(range(z_means.shape[0])):
            z = z_means.repeat(n_steps, 1)
            z[:, i] = torch.linspace(
                z_mins[i], z_maxs[i], n_steps).to(self._device)

            gen_verts = self._model.decoder(z.to(self._device))
            gen_verts = self._unnormalize_verts(gen_verts)

            differences_from_first = self._renderer.compute_vertex_errors(
                gen_verts, gen_verts[0].expand(gen_verts.shape[0], -1, -1))
            max_distances.append(differences_from_first[-1, ::])
            renderings = self._renderer.render(gen_verts).detach().cpu()
            all_renderings.append(renderings)
            differences_renderings = self._renderer.render(
                gen_verts, differences_from_first,
                error_max_scale=5).cpu().detach()
            all_rendered_differences.append(differences_renderings)
            frames = torch.cat([renderings, differences_renderings], dim=-1)
            all_frames.append(
                torch.cat([frames, torch.zeros_like(frames)[:2, ::]]))

        file_path = os.path.join(self._out_dir, 'latent_exploration.mp4')
        # write_video(file_path, torch.cat(all_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=4)
        frames_np = (torch.cat(all_frames, dim=0).permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        imageio.mimsave(file_path, frames_np, fps=4)

        # Same video as before, but effects of perturbing each latent variables
        # are shown in the same frame. Only error maps are shown.
        grid_frames = []
        grid_nrows = 2

        stacked_frames = torch.stack(all_rendered_differences)

        for i in range(stacked_frames.shape[1]):
            grid_frames.append(
                make_grid(stacked_frames[:, i, ::], padding=10,
                          pad_value=1, nrow=grid_nrows))
        save_image(grid_frames[-1],
                   os.path.join(self._out_dir, 'latent_exploration_tiled.png'))
        file_path = os.path.join(self._out_dir, 'latent_exploration_tiled.mp4')
        # write_video(file_path, torch.stack(grid_frames, dim=0).permute(0, 2, 3, 1) * 255, fps=1)
        frames_np = (torch.stack(grid_frames, dim=0).permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        imageio.mimsave(file_path, frames_np, fps=1)
        
        # Same as before, but only output meshes are used
        stacked_frames_meshes = torch.stack(all_renderings)
        grid_frames_m = []
        for i in range(stacked_frames_meshes.shape[1]):
            grid_frames_m.append(
                make_grid(stacked_frames_meshes[:, i, ::], padding=10,
                          pad_value=1, nrow=grid_nrows))
        file_path = os.path.join(self._out_dir, 'latent_exploration_outs_tiled.mp4')
        # write_video(file_path, torch.stack(grid_frames_m, dim=0).permute(0, 2, 3, 1) * 255, fps=4)
        frames_np = (torch.stack(grid_frames_m, dim=0).permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        imageio.mimsave(file_path, frames_np, fps=4)

        # Create a plot showing the effects of perturbing latent variables
        df = pd.DataFrame(columns=['mean_dist', 'z_var', 'latent_type'])
        df_row = 0
        for zi, vert_distances in enumerate(max_distances):
            mean_distance = torch.mean(vert_distances).item()
            if zi == 0:
                latent_type = "disease"
            elif zi == 1:
                latent_type = "age"
            else:
                latent_type = "identity"
            df.loc[df_row] = [mean_distance, zi, latent_type]
            df_row += 1

        sns.set_theme(style="ticks")
        palette = {"disease": "red", "age": "blue", "identity": "green"}

        # Separate data for points (disease, age) and line (identity)
        point_data = df[df['latent_type'].isin(['disease', 'age'])]
        line_data = df[df['latent_type'] == 'identity']

        # Plot points for disease and age latents
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=point_data, x="z_var", y="mean_dist", hue="latent_type", palette=palette, s=100, marker='o')

        # Plot line for identity latents
        sns.lineplot(data=line_data, x="z_var", y="mean_dist", hue="latent_type", palette=palette)

        # Adjust x-axis ticks to show values from 0 to 12, incrementing by 1, but stopping at 10
        plt.xticks(ticks=np.arange(0, 12, 1))

        plt.xlabel("Latent Variable Index")
        plt.ylabel("Mean Distance")
        plt.title("Shape Changes Across Latent Variables")
        plt.legend(title="Latent Type")
        plt.savefig(os.path.join(self._out_dir, 'latent_exploration.svg'))
        # plt.savefig(os.path.join(self._out_dir, 'latent_exploration.png'))
        print("Finished...")

    
    def age_encoder_decoder_accuracy(self, eval_loader_name=None):

        """
        
        This function encodes the subjects, changes the age latent to a random age, decodes, then encodes again and meaures if the second encoding can generate 
        the same age that it was assigned to after the first encoding. If the encoder check test shows the encoder works well, this tests will show how well the decoder performs.

        Output: two scatter plots. One for assigned age vs predicted age after second encoding and one for GT age vs predicted age after second encoding. Both the training and val/test set are plotted
        
        """

        for j in range(2):

            if j == 0:
                # data_loader = train_loader
                test_name = 'TRAIN'
                z_list = self._train_all_latents_list
                re_2_list = self._train_re2_age_list
                gt_age_list = self._train_gt_ages_list
            elif j == 1 and eval_loader_name=='VAL':
                # data_loader = eval_loader
                test_name = 'VAL'
                z_list = self._val_all_latents_list
                re_2_list = self._val_re2_age_list
                gt_age_list = self._val_gt_ages_list
            elif j == 1 and eval_loader_name=='TEST':
                # data_loader = eval_loader
                test_name = 'TEST'
                z_list = self._test_all_latents_list
                re_2_list = self._test_re2_age_list
                gt_age_list = self._test_gt_ages_list
            else:
                raise ValueError("eval_loader must be either 'VAL' or 'TEST'")

            age_latents_gt = []
            age_preds_encoder = []
            age_latents_rand = []
            age_preds_decoder = []
            id_latents_1 = []
            id_latents_2 = []

            for i in range(len(z_list)):

                id_latents_1.append(z_list[i][2:])

                # age_preds = re_2_list[i]
                age_preds_encoder.append(re_2_list[i])

                age_rand = random.randrange(0, 17)
                age_latents_rand.append(age_rand)
                z_age_latent_rand = self._age_to_z[age_rand]["z_value"]
                z_list[i][1] = z_age_latent_rand

                age_latents_gt.append(gt_age_list[i])

            z_tensor = torch.tensor(z_list, dtype=torch.float32, device=self._device)
            batch_size = 64  # tune based on your GPU
            with torch.no_grad():
                for start in range(0, z_tensor.shape[0], batch_size):
                    end = start + batch_size
                    z_batch = z_tensor[start:end].to(self._device)   

                    gen_verts = self._model.decoder(z_batch)            
                    _, z_2_batch, _, _, re_2_2_batch = self._model(gen_verts)

                    for k in range(z_2_batch.shape[0]):
                        id_latents_2.append(z_2_batch[k][2:].detach().cpu().numpy())
                        age_preds_decoder.append(re_2_2_batch[k].detach().cpu().numpy())
            
            # for both train and test, calculate MSE between id_latents_1 and id_latents_2 and save in .txt file
            
            id_latents_1 = np.array(id_latents_1)
            id_latents_2 = np.array(id_latents_2)
            mse_overall = np.mean((id_latents_1 - id_latents_2) ** 2)

            line_to_add_1 = f'{test_name}\n- MSE overall: {mse_overall}'
            filename = os.path.join(self._out_dir, 'results.txt')
            if not os.path.exists(filename):
                with open(filename, 'w') as file:
                    file.write('')
            else:
                print(f"{filename} already exists.")
            with open(filename, 'a') as file:
                file.write('Identity preservation test results:\n')
                file.write(line_to_add_1)
                file.write('\n' * 2)

        ### PLOT RESULTS ###

            # Convert age_random_all and age_predict_all to numpy arrays for easier manipulation
            age_latents_gt = np.array(age_latents_gt)
            age_preds_encoder = np.array(age_preds_encoder)
            age_latents_rand = np.array(age_latents_rand)
            age_preds_decoder = np.array(age_preds_decoder)

            for i in range(2):

                if i == 0:
                    test_name = 'encoder'
                    age_latents = age_latents_gt.reshape(-1).astype(np.float32)
                    age_preds = age_preds_encoder.reshape(-1).astype(np.float32)
                else:
                    test_name = 'decoder'
                    age_latents = age_latents_rand.reshape(-1).astype(np.float32)
                    age_preds = age_preds_decoder.reshape(-1).astype(np.float32)

                plt.figure(figsize=(8, 6))
                plt.clf()

                colors = plt.cm.tab10.colors 

                # containers for per-age MAE
                max_age_int = 17
                per_age_abs_errors = [[] for _ in range(max_age_int + 1)]
                
                # Calculate MAE for the current latent
                abs_err = np.abs(np.array(age_preds) - np.array(age_latents))
                mae = np.mean(abs_err)

                # mse = np.mean((np.array(age_preds) - np.array(age_latents))**2)
                # mse_per_latent.append(mse)

                # --- NEW: accumulate absolute errors per integer GT age bin ---
                gt_int = np.clip(np.floor(age_latents).astype(int), 0, max_age_int)
                for age_bin in range(max_age_int + 1):
                    mask = (gt_int == age_bin)
                    if np.any(mask):
                        per_age_abs_errors[age_bin].extend(abs_err[mask].tolist())
                # --------------------------------------------------------------
                
                # Plot the scatter for the current latent
                plt.scatter(age_latents, age_preds, color=colors[1 % len(colors)], label=f'MAE: {mae:.4f}', alpha=0.7)

                # Add a diagonal line for reference
                plt.plot([0, 17], [0, 17], 'r--', label='Ideal')  # Assuming age range is 0-18

                # Set ticks
                plt.xticks(range(0, 18))
                plt.yticks(range(0, 18))

                if i == 0:
                    plt.title('Encoder: Predicted vs Ground Truth for Each Age Latent')
                    plt.xlabel('Ground Truth Age (years)')
                    plt.ylabel('Predicted Age Latent Value')
                else:
                    plt.title('Decoder: Predicted vs Randomly Assigned for Each Age Latent')
                    plt.xlabel('Randomly Assigned Age (years)')
                    plt.ylabel('Predicted Age Latent Value')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
                plt.grid(True)

                # --- NEW: compute and display per-integer-age MAE block ---
                per_age_mae_lines = []
                for age_bin in range(max_age_int + 1):
                    if len(per_age_abs_errors[age_bin]) > 0:
                        age_mae = np.mean(per_age_abs_errors[age_bin])
                        per_age_mae_lines.append(f'Age {age_bin} (MAE: {age_mae:.4f})')
                    else:
                        per_age_mae_lines.append(f'Age {age_bin} (MAE: -)')

                if per_age_mae_lines:
                    per_age_text = "MAE per age (years):\n" + "\n".join(per_age_mae_lines)

                    plt.text(1.02, 0.42, per_age_text, transform=plt.gca().transAxes, fontsize=10, fontfamily='sans-serif', color='black', va='top',
                        bbox=dict(
                            boxstyle='round',
                            facecolor='white',
                            edgecolor='black',
                            alpha=0.8
                        )
                    )
            
                # Save the plot
                if j == 0:
                    test_name =  test_name + '_train'
                else:
                    test_name = test_name + '_eval'
                # file_path = os.path.join(self._out_dir, f'{test_name}_accuracy_scatter_plot.png')
                file_path_svg = os.path.join(self._out_dir, f'{test_name}_accuracy_scatter_plot.svg')
                # plt.savefig(file_path, bbox_inches='tight')  
                plt.savefig(file_path_svg, bbox_inches='tight')  

    def set_seed(self, seed):

        """
        
        This function makes sure all random operation produce the same results each time the code is run.
        
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
            
    
    def age_prediction_MLP(self, eval_loader_name=None):
        """
        This function trains a MLP model to predict the age of the subjects based on the feature latents. 

        If disentanglement is successful, the model should NOT be able to predict the age of the subjects based on the feature latents.

        Output: plot of training loss and scatter plot of predicted age against ground truth age
        """

        train_id_latents = self._train_id_latents_list
        train_gt_age = self._train_gt_ages_list
        if eval_loader_name == 'VAL':
            eval_id_latents = self._val_id_latents_list
            eval_gt_age = self._val_gt_ages_list
        elif eval_loader_name == 'TEST':
            eval_id_latents = self._test_id_latents_list
            eval_gt_age = self._test_gt_ages_list
        else:
            raise ValueError("eval_loader_name must be either 'VAL' or 'TEST'")
        
        sc = StandardScaler()
        train_id_latents_scaled = sc.fit_transform(train_id_latents)
        eval_id_latents_scaled = sc.transform(eval_id_latents)

        train_id_latents = torch.tensor(train_id_latents_scaled, dtype=torch.float32)
        eval_id_latents = torch.tensor(eval_id_latents_scaled, dtype=torch.float32)

        train_gt_age_tensor = torch.tensor(train_gt_age, dtype=torch.float32).view(-1, 1)
        eval_gt_age_tensor = torch.tensor(eval_gt_age, dtype=torch.float32).view(-1, 1)

        self.set_seed(42)

        train_dataset = TensorDataset(train_id_latents, train_gt_age_tensor)
        train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True)

        input_size = train_id_latents.shape[1]

        # Define the MLP model, loss function, and optimizer
        model = nn.Sequential(
            nn.Linear(input_size, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)) 
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        def train_model(model, criterion, optimizer, dataloader, epochs=100):
            model.train()
            losses = []
            for epoch in range(epochs):
                for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
                losses.append(loss.item())
                # self.log['test/MLP_loss'].log(loss.item())
            return losses

        losses = train_model(model, criterion, optimizer, train_loader)

        # Plot the losses
        plt.figure()
        plt.clf()
        plt.plot(losses)
        plt.title('MLP Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        file_path = os.path.join(self._out_dir, 'mlp_training_loss.svg')
        plt.savefig(file_path)
        # self.log['test/mlp_training_loss'].upload(file_path)

        # Evaluate the model
        def evaluate_model(model, X, y):
            model.eval()
            with torch.no_grad():
                predictions = model(X).view(-1)
                mae = torch.mean(torch.abs(predictions - y.squeeze_(1)))
            return predictions.numpy(), mae.item()

        train_ages_pred, train_mean_age_diff = evaluate_model(model, train_id_latents, train_gt_age_tensor)
        eval_ages_pred, eval_mean_age_diff = evaluate_model(model, eval_id_latents, eval_gt_age_tensor)

        # Plot the results
        age_range = '0-17'
        min_age, max_age = map(int, age_range.split('-'))

        # Define base marker size
        base_marker_size = 50

        plt.figure(figsize=(6, 6))
        plt.clf()

        # Scatter plot with fixed marker sizes
        plt.scatter(train_gt_age, train_ages_pred, s=base_marker_size, color='yellow', marker='x', label='Train dataset')
        plt.scatter(eval_gt_age, eval_ages_pred, s=base_marker_size, color='green', marker='o', label='Eval dataset')
        plt.plot([0, max_age], [0, max_age], 'r--')

        # Add title, labels, and text
        plt.title('Age prediction on feature latents')
        plt.xlabel('Ground truth age (years)')
        plt.ylabel('Predicted age (years)')
        plt.text(0.25, 0.1, f'Mean absolute difference (train) = {round(train_mean_age_diff, 2)} years', transform=plt.gca().transAxes)
        plt.text(0.25, 0.05, f'Mean absolute difference (val) = {round(eval_mean_age_diff, 2)} years', transform=plt.gca().transAxes)

        # Fixed marker sizes for legend
        legend_handles = [
            plt.scatter([], [], color='yellow', marker='x', s=base_marker_size, label='Train dataset'),
            plt.scatter([], [], color='green', marker='o', s=base_marker_size, label='Eval dataset')
        ]
        plt.legend(handles=legend_handles, loc='upper left')

        # Set ticks
        plt.xticks(range(0, 18))
        plt.yticks(range(0, 18))

        # file_path = os.path.join(self._out_dir, f'mlp_age_prediction_{age_range}.png')
        file_path_svg = os.path.join(self._out_dir, f'mlp_age_prediction_{age_range}.svg')
        # plt.savefig(file_path)
        plt.savefig(file_path_svg)
        # self.log['test/mlp_age_prediction'].upload(file_path)

        # Create a clean plot for the test set
        plt.figure(figsize=(6, 6))
        plt.clf()

        # Scatter plot for test set only
        plt.scatter(eval_gt_age, eval_ages_pred, s=base_marker_size, color='green', marker='o')
        plt.plot([0, max_age], [0, max_age], 'r--')

        # Add title with MAE
        plt.title(f'Test Set Age Prediction (MAE = {round(eval_mean_age_diff, 5)} years)')

        # Add axis labels
        plt.xlabel('Ground truth age (years)')
        plt.ylabel('Predicted age (years)')

        # Set ticks
        plt.xticks(range(0, 18))
        plt.yticks(range(0, 18))

        # file_path_clean = os.path.join(self._out_dir, 'mlp_age_prediction_clean.png')
        file_path_clean_svg = os.path.join(self._out_dir, 'mlp_age_prediction_clean.svg')
        # plt.savefig(file_path_clean)
        plt.savefig(file_path_clean_svg)

    
    def age_latent_changing(self, eval_loader_name=None):

        """
        
        This function generates an image of 4 subjects from a batch and changes the age latent to 5 differenet ages and displays them with their difference maps compared to the orginal mesh. 
        A subjects goes through the encoder, the age latent is changed, then goes through the decoder. 

        10 different tests are run. Changing the age for each feature (9) and then changing the age for all features (1).

        Output: 10 images of a batch with 5 different ages & difference maps when compared to the:
         - first age (0)

        """

        padding_value = 10

        age_latent_ranges = [0.00, 4.00, 8.00, 12.00, 17.00] 
        z_age_latent_ranges = [self._age_to_z[age]["z_value"] for age in age_latent_ranges]

        if eval_loader_name == 'VAL':
            batch_latents = self._val_all_latents_list[0:4]
            batch_gt_ages = self._val_gt_ages_list[0:4]
        elif eval_loader_name == 'TEST':        
            batch_latents = self._test_all_latents_list[0:4]
            batch_gt_ages = self._test_gt_ages_list[0:4]
        else:
            raise ValueError("eval_loader_name must be either 'VAL' or 'TEST'")

        error_scale = 5

        latents_tensor = torch.tensor(batch_latents, dtype=torch.float32)  

        batch_latents_extended = latents_tensor.unsqueeze(1).repeat(1, len(z_age_latent_ranges), 1)

        ages_tensor = torch.tensor(z_age_latent_ranges, dtype=torch.float32, device=batch_latents_extended.device)

        batch_latents_extended[:, :, 1] = ages_tensor.view(1, len(z_age_latent_ranges))

        z = batch_latents_extended.reshape(-1, latents_tensor.shape[1])

        with torch.no_grad():
            gen_verts = self._model.decoder(z.to(self._device))
        gen_verts = self._unnormalize_verts(gen_verts)

        renderings = self._renderer.render(gen_verts).cpu()

        output = []
        for j in range(len(batch_latents)):

            output.extend(renderings[j*(len(z_age_latent_ranges)):(j+1)*(len(z_age_latent_ranges))])
            first_index = j*(len(z_age_latent_ranges))

            for k in range(len(z_age_latent_ranges)):

                k_index = first_index + k

                differences_from_first = self._renderer.compute_vertex_errors(gen_verts[k_index].unsqueeze(0), gen_verts[first_index].unsqueeze(0))
                differences_renderings_first = self._renderer.render(gen_verts[k_index].unsqueeze(0), differences_from_first, error_max_scale=error_scale).cpu().detach()
                output.append(differences_renderings_first.squeeze())

        # create image

        stacked_frames = torch.stack(output)
        file_path = os.path.join(self._out_dir, f'age_latent_changing.png')
        grid = make_grid(stacked_frames, padding=padding_value, pad_value=255, nrow=len(z_age_latent_ranges)) 
        save_image(grid, file_path)

        line_to_add = 'age_latent_changing original ages: ' + str(batch_gt_ages)
        filename = os.path.join(self._out_dir, 'results.txt')

        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write('')
        else:
            print(f"{filename} already exists.")

        with open(filename, 'a') as file:
            file.write(line_to_add)
            file.write('\n' * 2)


    def tsne_visualization(self):
        """
        This function performs t-SNE on the identity latents and visualizes 
        their clustering based on age.
        """

        random_state = 42
        self.set_seed(random_state)
        cmap='viridis'

        subset_latents = np.concatenate((self._train_id_latents_list, self._val_id_latents_list, self._test_id_latents_list), axis=0)
        gt_feature = np.concatenate((self._train_gt_ages_list, self._val_gt_ages_list, self._test_gt_ages_list), axis=0)
        datasets = np.concatenate((self._train_dataset_list, self._val_dataset_list, self._test_dataset_list), axis=0)

        sc = StandardScaler()
        subset_latents_scaled = sc.fit_transform(subset_latents)

        tsne = TSNE(n_components=2, random_state=random_state)
        tsne_results = tsne.fit_transform(subset_latents_scaled)

        ### PLOT AGE ###
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=gt_feature, cmap=cmap, alpha=0.6)

        plt.colorbar(scatter, label='Age')
        plt.title('t-SNE Visualization of Identity Latents by Age')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')

        file_path = os.path.join(self._out_dir, 'tsne_identity_latents_vs_age.svg')
        plt.savefig(file_path)
        plt.close()

        ### PLOT DATASET ###
        gt_feature_str = datasets
        unique_values = np.unique(gt_feature_str)
        value_to_number = {value: idx for idx, value in enumerate(unique_values)}
        gt_feature_numeric = np.array([value_to_number[value] for value in gt_feature_str])
        gt_feature = gt_feature_numeric
        cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(unique_values))))

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=gt_feature, cmap=cmap, alpha=0.6)

        cbar = plt.colorbar(scatter, ticks=range(len(unique_values)), label='Datasets')
        cbar.ax.set_yticklabels(unique_values)
        plt.title('t-SNE Visualization of Identity Latents by Dataset')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')

        file_path = os.path.join(self._out_dir, 'tsne_identity_latents_vs_dataset.svg')
        plt.savefig(file_path)
        plt.close()


    def compute_r2(self, X_tr, Y_tr, X_te, Y_te):

        """
        Compute R², works for 1 or many age latents.

        """
        
        model = LinearRegression()
        model.fit(X_tr, Y_tr)
        Y_pred = model.predict(X_te)
        r2 = r2_score(Y_te, Y_pred, multioutput='variance_weighted')

        return float(r2)


    def stats_tests_correlation(self):

        """
        Perform statistical tests to check if age is disentangled from feature latents.

        """

        all_latents_train_val = np.concatenate((self._train_all_latents_list, self._val_all_latents_list), axis=0).astype(np.float32)
        identity_latents_train_val = np.concatenate((self._train_id_latents_list, self._val_id_latents_list), axis=0).astype(np.float32)
        identity_latents_test = np.asarray(self._test_id_latents_list, dtype=np.float32)
        age_latents_train_val = all_latents_train_val[:, 1:2] 
        age_latents_test = np.array(self._test_all_latents_list, dtype=np.float32)[:, 1:2]
        gt_ages_train_val = np.concatenate((self._train_gt_ages_list, self._val_gt_ages_list), axis=0).astype(np.float32)
        gt_ages_test = np.asarray(self._test_gt_ages_list, dtype=np.float32)

        # ------------------------------------------------------------------
        # 1) Cross-latent global SAP (age in id / id in age)
        # ------------------------------------------------------------------

        sap_score = _compute_sap(identity_latents_train_val.T, age_latents_train_val.T, identity_latents_test.T, age_latents_test.T, continuous_factors=True)
        print("SAP (age latents info in identity latents):", sap_score)

        r2_id_given_age = self.compute_r2(age_latents_train_val, identity_latents_train_val, age_latents_test, identity_latents_test)
        print(f"R² (identity latents info in age latent): {r2_id_given_age}")


        # ----------------------------------------------------------------------------
        # 2) Proper SAP: identity latents vs *ground-truth age* (using diagonal only)
        # ----------------------------------------------------------------------------

        sap_age_in_id_gt = _compute_sap(
            identity_latents_train_val.T,       
            gt_ages_train_val.T,    
            identity_latents_test.T,             
            gt_ages_test.T,           
            continuous_factors=True,
        )
        print("SAP (GT age in identity latents):", sap_age_in_id_gt)

        r2_age_vs_latent = self.compute_r2(age_latents_train_val, gt_ages_train_val, age_latents_test, gt_ages_test)
        print(f"R² (GT age in age latent): {r2_age_vs_latent}")

        # ------------------------------------------------------------------
        # 3) HIPPOCAMPUS PAPER SAP implementation
        # ------------------------------------------------------------------

        ages_np = gt_ages_train_val[:, 0]
        lat_np = all_latents_train_val 

        # Per-dimension Pearson correlation with gt age
        corr_per_dim = [stats.pearsonr(ages_np, lat_np[:, d])[0] for d in range(lat_np.shape[1])]

        # SAP score for age (continuous factor)
        sap_score_hipp = sap(factors=ages_np[:, None], codes=lat_np, continuous_factors=True, regression=True)

        # How much age leaks into the other latent dims (max |corr| excluding age_dim)
        leakage_age_into_others = 0.0

        latent_size = 12
        disease_latent_size = 1
        age_latent_size = 1
        identity_start = disease_latent_size + age_latent_size 
        identity_end = latent_size                             

        if lat_np.shape[1] > 1:
            leakage_age_into_others = max(abs(c) for i, c in enumerate(corr_per_dim) if identity_start <= i < identity_end)

        print("Per-dim Pearson r (GT_age in all_latents):", np.round(corr_per_dim, 3))
        print(f"Hippocampus SAP (GT_age in all_latents): {sap_score_hipp}")
        print(f"GT_age leakage into identity latents (max |corr| excluding best): {leakage_age_into_others}\n")
    
        output_file = os.path.join(self._out_dir, 'dis_stats.txt')
        with open(output_file, 'w') as f:
            f.write(f"SAP (age latents info in identity latents): {sap_score}\n\n")
            f.write(f"R² (identity latents info in age latent): {r2_id_given_age} \n\n")
            f.write(f"SAP (GT age in identity latents): {sap_age_in_id_gt}\n\n")
            f.write(f"R² (GT age in age latent): {r2_age_vs_latent} \n\n")
            f.write(f"Per-dim Pearson r (GT_age in all_latents): {np.round(corr_per_dim, 3)}\n\n")
            f.write(f"Hippocampus SAP (GT_age in all_latents): {sap_score_hipp} \n\n")
            f.write(f"GT_age leakage into identity latents (max |corr| excluding best): {leakage_age_into_others}\n\n")


    def proportions(self, eval_loader_name):
        """
        This function takes the first batch from the input data, passes them through the encoder,
        changes the age latents to all equal the same value for all integer ages between 'age_range',
        and then passes all these changed ages data through the generator.
        """

        self.set_seed(42)

        if eval_loader_name == 'VAL':
            gt_ages = self._val_gt_ages_list
            z_list = self._val_all_latents_list
            fname_list = self._val_fname_list
        elif eval_loader_name == 'TEST':
            gt_ages = self._test_gt_ages_list
            z_list = self._test_all_latents_list
            fname_list = self._test_fname_list
        else:
            raise ValueError("eval_loader_name must be either 'VAL' or 'TEST'")

        age_lower = 0
        age_upper = 17

        all_gen_verts = []
        all_mesh_names = []

        z_tensor = torch.tensor(z_list, dtype=torch.float32, device=self._device)
        batch_size = 64  # tune based on your GPU
        with torch.no_grad():
            for start in range(0, z_tensor.shape[0], batch_size):
                end = start + batch_size
                z_batch = z_tensor[start:end].to(self._device)

                for age in range(age_lower, age_upper + 1):
                    age_latent_value = self._age_to_z[age]["z_value"]
                    z_batch[:, 1:2] = age_latent_value

                    gen_verts = self._model.decoder(z_batch)
                    gen_verts = self._unnormalize_verts(gen_verts)

                    mesh_names = [f"{fname}_{age}" for fname in fname_list[start:end]]
                    all_gen_verts.append(gen_verts)
                    all_mesh_names.append(mesh_names)

        all_gen_verts = torch.cat(all_gen_verts, dim=0)
        all_mesh_names = [item for sublist in all_mesh_names for item in sublist]

        template_path = self._template_path
        dataset_metadata_path = self._dataset_metadata_path
        output_directory = self._out_dir
        folder_path = None
        dataset_type = 'friday_unified'
        calculate_distances_in_folder(folder_path, template_path, all_gen_verts, all_mesh_names, dataset_type, output_directory)
        add_proportions_age_gender_to_csv(folder_path, dataset_type, output_directory, dataset_metadata_path)
        distance_proportion_averages(dataset_type, output_directory)

    def plot_proportions(self):
        output_directory = self._out_dir
        dataset_type = 'friday_unified'

        # model output proportions
        csv_path1 = os.path.join(output_directory, f"{dataset_type}_proportion_averages.csv")
        df1 = pd.read_csv(csv_path1)

        measurements_path = '/home/athena/COMPARISON_Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/tests/measurements'

        # dataset proportions
        csv_path2 = os.path.join(measurements_path, f"{dataset_type}_proportion_averages.csv")
        df2 = pd.read_csv(csv_path2)

        # farkas proportions
        csv_path3 = f"{measurements_path}/farkas_proportion_averages.csv"
        df3 = pd.read_csv(csv_path3)

        # facebase proportions
        csv_path4 = f"{measurements_path}/facebase_proportion_averages.csv"
        df4 = pd.read_csv(csv_path4)

        age_range = "0-17"
        age_lower, age_upper = map(int, age_range.split('-'))
        df2 = df2[(df2['age'] >= age_lower) & (df2['age'] <= age_upper)]
        df3 = df3[(df3['age'] >= age_lower) & (df3['age'] <= age_upper)]
        df4 = df4[(df4['age'] >= age_lower) & (df4['age'] <= age_upper)]

        proportion_columns = ['n-sto:n-gn', 'n-sto:sto-gn', 'sto-gn:n-gn', 'zy_right-zy_left:go-right-go-left']
        
        # Plot the data
        def plot_comparison(df_a, df_b, proportion_name, label_a, label_b, output_directory, mse):
            plt.figure(figsize=(10, 6))
            colors = {'male': 'blue', 'female': 'green'}
            for gender in df_a['gender'].unique():
                gender_data_a = df_a[df_a['gender'] == gender]
                gender_data_b = df_b[df_b['gender'] == gender]
                plt.plot(gender_data_a['age'], gender_data_a[proportion_name], label=f'{gender} ({label_a})', linestyle='-', color=colors[gender])
                plt.plot(gender_data_b['age'], gender_data_b[proportion_name], label=f'{gender} ({label_b})', linestyle='--', color=colors[gender])
            plt.xlabel('Age', fontsize=22)
            plt.ylabel('Proportion Value', fontsize=22)
            plt.title(f'{label_a} vs {label_b}', fontsize=26)
            plt.suptitle(f'Proportion {proportion_name}', fontsize=10)
            if mse is not None:
                plt.text(1.05, 1.05, f'MSE: {mse}', fontsize=18, color='black', transform=plt.gca().transAxes, ha='left', va='top')
            plt.legend(fontsize=22, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.xticks(range(0, 18), fontsize=22)
            plt.yticks(fontsize=22)
            file_path = os.path.join(output_directory, f'proportions_{proportion_name}_{label_a}_vs_{label_b}.svg')
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

        # Plot each proportion for the three comparisons
        for proportion in proportion_columns:
            # Farkas vs. FaceBase
            mse = None
            plot_comparison(df3, df4, proportion, 'Farkas', 'FaceBase', output_directory, mse)

            # FaceBase vs. Dataset
            mse = None
            plot_comparison(df4, df2, proportion, 'FaceBase', 'Dataset', output_directory, mse)

            # Dataset vs. Model
            # mse = calculate_mse(df2, df1, proportion)
            s1 = df1[proportion]
            s2 = df2[proportion]
            mask = ~(s1.isna() | s2.isna())
            s1_clean = s1[mask]
            s2_clean = s2[mask]
            mse = mean_squared_error(s2_clean, s1_clean)
            plot_comparison(df2, df1, proportion, 'Dataset', 'Model', output_directory, mse)


if __name__ == '__main__':

    model_directory = "/raid/compass/athena/outputs/sc-vae-redo"
    output_directory = f"{model_directory}/outputs"
    # checkpoint_dir = os.path.join(model_directory, 'checkpoints')
    base_path = f"{model_directory}/models_contrastive_inhib_decrease_trainset_tc/10"
    # base_path = f"/raid/compass/athena/data/PLY_friday_unified_meshes_subset_0_17/raw/torus/models_contrastive_inhib_decrease_trainset_tc/10"

    # make output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    configurations = None
    logging = None
    normalization_dict = None
    manager = None

    device = torch.device('cuda', 0)

    # Load the saved model
    model_state_dict = torch.load(f"{base_path}/model_state_dict.pt")
    in_channels = torch.load(f"{base_path}/in_channels.pt")
    out_channels = torch.load(f"{base_path}/out_channels.pt")
    latent_channels = torch.load(f"{base_path}/latent_channels.pt")
    spiral_indices_list = torch.load(f"{base_path}/spiral_indices_list.pt")
    up_transform_list = torch.load(f"{base_path}/up_transform_list.pt")
    down_transform_list = torch.load(f"{base_path}/down_transform_list.pt")
    std = torch.load(f"{base_path}/std.pt")
    mean = torch.load(f"{base_path}/mean.pt")
    template_face = torch.load(f"{base_path}/faces.pt")

    # Create an instance of the model
    model = AE(in_channels, out_channels, latent_channels,
            spiral_indices_list, down_transform_list,
            up_transform_list)
    model.load_state_dict(model_state_dict)
    model.to(device)
    # Set the model to evaluation mode
    model.eval()

    dataset_metadata_path = "/home/athena/3DVAE-AgeDisentangled/preprocessing_data/friday_all_datasets.csv"
    template_fp = "/raid/compass/athena/data/PLY_friday_unified_meshes_subset_0_17/template/template.ply"
    data_fp = "/raid/compass/athena/data/PLY_friday_unified_meshes_subset_0_17"
    test_exp = "bareteeth"
    split = "interpolation"

    meshdata = MeshData(data_fp,
                        template_fp,
                        split=split,
                        test_exp=test_exp)

    train_loader = DataLoader(meshdata.train_dataset, batch_size=4)
    val_loader = DataLoader(meshdata.val_dataset, batch_size=4)
    test_loader = DataLoader(meshdata.test_dataset, batch_size=4)

    tester = Tester(manager, normalization_dict, train_loader, val_loader, test_loader,
                    output_directory, configurations, logging, model, device, meshdata, dataset_metadata_path)

    tester()



