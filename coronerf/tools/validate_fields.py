#########################################################
### validation scripts for fields
#########################################################

'''
some functions from this file require pycelp, which is dependent on OS during runtime
only import after detecting runtime and setting the environment variables to link CHIANTI (diff for windows vs linux)
'''
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from ..util.torch import to_numpy
from ..experiments.dirs import spherical_debug, regular_debug
from ..geom.coords import llr_to_xyz
from ..fields.interp_util import match_field
from .atomic import compute_ccoef

def test_temp_field(pipeline):
    '''
    basic unit test for the temperature trilinear interpolator
    '''
    # stuff we need
    dirs = pipeline.dirs
    temp_field = pipeline.temp_field


    single_point_test = True
    periodic_test = True
    lg = temp_field.logger

    with spherical_debug(temp_field):
        # single point test
        if single_point_test:
            lg.debug(f'Beginning single point test...')

            lon_i, phi_i, r_i = 167, 67, 106

            # llr
            x_llr = torch.tensor([temp_field.lon_axis[lon_i], 
                                    temp_field.phi_axis[phi_i], 
                                    temp_field.r_axis[r_i]])
            
            x_xyz = llr_to_xyz(x_llr)[None, ...] # R_sun

            lg.debug(f'llr coord: {to_numpy(x_llr)}')
            lg.debug(f'xyz coord: {to_numpy(x_xyz)}')

            x_xyz = match_field(temp_field.field, x_xyz)
            
            val_pred = temp_field.forward(x_xyz)
            val_gt = temp_field.field[0,0,r_i,phi_i,lon_i]
            
            lg.debug(f'pred: {to_numpy(val_pred)[0]:.16f}')
            lg.debug(f'gt: {to_numpy(val_gt):.16f}')

        # test periodic smoothness of interpolator
        if periodic_test:
            lg.debug(f'Beginning periodic test...')

            r_i = 167
            phi_skip = 5
            lon_ext = 20

            #gt field
            val_gt = torch.cat((temp_field.field[0, 0, r_i, ::phi_skip, -lon_ext:], 
                                temp_field.field[0, 0, r_i, ::phi_skip, 0:lon_ext]),
                                axis = -1)
            
            num_phi, num_lon = val_gt.shape # dimensions
            val_gt = val_gt.view(-1) # flatten
            
            lg.debug(f'val_gt shape: {val_gt.shape}')
            
            # create coordinates
            lon_llr = torch.cat((temp_field.lon_axis[-lon_ext:], 
                                 temp_field.lon_axis[:lon_ext]))
            phi_llr = temp_field.phi_axis[::phi_skip]
            r_llr = temp_field.r_axis[r_i:r_i+1] # preserves dtype and device

            lg.debug(f'lon_llr shape: {lon_llr.shape}, type: {lon_llr.dtype}')
            lg.debug(f'phi_llr shape: {phi_llr.shape}, type: {phi_llr.dtype}')
            lg.debug(f'r_llr shape: {r_llr.shape}, type: {r_llr.dtype}')

            # mesh grid
            r_llr_mg, phi_llr_mg, lon_llr_mg = torch.meshgrid(r_llr, phi_llr, lon_llr, indexing = 'ij')

            lg.debug(f'lon_llr_mg shape: {lon_llr_mg.shape}')
            lg.debug(f'phi_llr_mg shape: {phi_llr_mg.shape}')
            lg.debug(f'r_llr_mg shape: {r_llr_mg.shape}')

            #lg.debug(f'lon_llr_mg contiguous: {lon_llr_mg.is_contiguous()}')
            #lg.debug(f'phi_llr_mg shape: {phi_llr_mg.is_contiguous()}')
            #lg.debug(f'r_llr_mg shape: {r_llr_mg.is_contiguous()}')

            # stack, flatten
            x_llr = torch.stack((lon_llr_mg.contiguous().view(-1), 
                                    phi_llr_mg.contiguous().view(-1), 
                                    r_llr_mg.contiguous().view(-1)), axis = 0)
            
            # make samples on rows dim
            x_llr = torch.transpose(x_llr, 0, 1)

            lg.debug(f'x_llr shape: {x_llr.shape}')

            x_xyz = llr_to_xyz(x_llr)

            lg.debug(f'x_xyz shape: {x_xyz.shape}, x_xyz type: {x_xyz.dtype}')

            # compute estimated values
            val_pred = temp_field.forward(x_xyz)

            lg.debug(f'val_pred shape: {val_pred.shape}')

            # compare
            frac_err = (val_pred - val_gt) / val_gt
            frac_err = to_numpy(frac_err)
            lg.debug(f'frac_err mean: {np.mean(frac_err)}')
            lg.debug(f'frac_err sigma: {np.std(frac_err)}')

            val_gt = to_numpy(val_gt.view(num_phi, num_lon))
            val_pred = to_numpy(val_pred.view(num_phi, num_lon))

            vmin = min(val_gt.min(), val_pred.min())
            vmax = max(val_gt.max(), val_pred.max())

            # plot

            # dont plot if no save
            if getattr(pipeline, "no_save", False):
                lg.debug("io.no_save = True -> skipping temp field test save")
                return

            fig, axs = plt.subplots(1, 2, figsize = (12, 4))

            # extended lon just for plotting
            lon_plot = lon_llr.clone()
            # everything that "wrapped" (smaller than first element) gets +2π
            wrap_mask = lon_plot < lon_plot[0]
            lon_plot[wrap_mask] += 2 * np.pi

            phi_min, phi_max = to_numpy(phi_llr[0]), to_numpy(phi_llr[-1])
            lon_min, lon_max = to_numpy(lon_plot[0]), to_numpy(lon_plot[-1])

            #lg.debug(f'lon_min, lon_max: {lon_min, lon_max}')

            plt.suptitle(f'T periodic reconstruction at r = {temp_field.r_axis[r_i]:.4f} R_sun')
            mappable = axs[0].imshow(val_gt, 
                                        vmin = vmin, 
                                        vmax = vmax,
                                        origin = 'lower',
                                        extent = [lon_min, lon_max, phi_min, phi_max],
                                        aspect = 'auto')
            plt.colorbar(mappable, ax = axs[0])
            axs[0].set_title('GT')

            mappable = axs[1].imshow(val_pred, 
                                        vmin = vmin, 
                                        vmax = vmax,
                                        origin = 'lower',
                                        extent = [lon_min, lon_max, phi_min, phi_max],
                                        aspect = 'auto')
            plt.colorbar(mappable, ax= axs[1])
            axs[1].set_title('Pred')

            for ax in axs:
                # choose tick positions in unwrapped space
                xticks = np.linspace(lon_min, lon_max, 5)
                yticks = np.linspace(phi_min, phi_max, 5)

                # map tick *labels* back to [0, 2π)
                xlabels = (xticks % (2 * math.pi))
                ylabels = yticks  # phi doesn’t wrap here

                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
                ax.set_xticklabels([f'{x:.2f}' for x in xlabels])
                ax.set_yticklabels([f'{y:.2f}' for y in ylabels])

                ax.set_ylabel('Phi (rad)')
                ax.set_xlabel('Lon (rad)')

            plt.tight_layout()

            subdir = dirs.figs / 'interp_test'
            subdir.mkdir(exist_ok=True, parents=True)

            fig_path = subdir / f"temp_periodic_r{r_i}.png"
            plt.savefig(fig_path, dpi=200, bbox_inches="tight")
            lg.info(f"Saved periodic temp debug figure to {fig_path}")

def test_ne_field(pipeline):
    '''
    basic unit test for the ne trilinear interpolator
    '''
    # stuff we need
    ne_field = pipeline.ne_field


    single_point_test = True
    lg = ne_field.logger

    with spherical_debug(ne_field):
        if single_point_test:
            lg.debug(f'Beginning single point test...')

            lon_i, phi_i, r_i = 167, 67, 106

            # llr
            x_llr = torch.tensor([ne_field.lon_axis[lon_i], 
                                  ne_field.phi_axis[phi_i], 
                                  ne_field.r_axis[r_i]])
            
            x_xyz = llr_to_xyz(x_llr)[None, ...] # R_sun

            lg.debug(f'llr coord: {to_numpy(x_llr)}')
            lg.debug(f'xyz coord: {to_numpy(x_xyz)}')

            x_xyz = match_field(ne_field.field, x_xyz)
            
            val_pred = ne_field.forward(x_xyz)
            val_gt = ne_field.field[0,0,r_i,phi_i,lon_i]
            
            lg.debug(f'pred: {to_numpy(val_pred)[0]:.16f}')
            lg.debug(f'gt: {to_numpy(val_gt):.16f}')

def test_ccoef_field(pipeline):
    '''
    basic unit test for the temperature trilinear interpolator

    was made very early for Fe XIII line only
    '''
    # dont run if its not fe13
    requested_wavelengths = list(
        getattr(pipeline, 'selected_wavelengths', None)
        or pipeline.cfg.get('ion_kwargs', {}).get('wavelengths', [])
        or []
    )
    if requested_wavelengths != [10747, 10798]:
        lg.warning(
            f'Skipping exact ccoef midpoint validation b/c validate_fields.py still '
            f'hard-wired to only Fe XIII pair, but got {requested_wavelengths}'
        )
        return

    # stuff we need
    ccoef_field = pipeline.ccoef_field
    dirs = pipeline.dirs


    single_point_test = True
    error_test = True
    lg = ccoef_field.logger

    with regular_debug(ccoef_field):
        if single_point_test:
            lg.debug(f'Beginning single point test...')

            ne_i, temp_i, r_i = 50, 50, 16 # x, y, z

            x_xyz = torch.tensor([ccoef_field.x_axis[ne_i], 
                                  ccoef_field.y_axis[temp_i], 
                                  ccoef_field.z_axis[r_i]])
            
            lg.debug(f'xyz coord: {to_numpy(x_xyz)}')

            x_xyz = match_field(ccoef_field.field, x_xyz)
            
            val_pred = ccoef_field.forward(x_xyz)
            val_gt = ccoef_field.field[0,:,r_i,temp_i,ne_i]
            
            lg.debug(f'pred: {to_numpy(val_pred)}')
            lg.debug(f'gt: {to_numpy(val_gt)}')
        if error_test:
            lg.debug(f'Beginning error test...')
            # x = ne, y = temp, z = r

            # build midpt coordinates
            Nx, Ny, Nz = ccoef_field.x_axis.shape[0], ccoef_field.y_axis.shape[0], ccoef_field.z_axis.shape[0]
            lg.debug(f'ccoef field shape (Nx, Ny, Nz): {Nx, Ny, Nz}')
            x_arr = ccoef_field.x_axis[:-1] + (ccoef_field.x_axis[1:] - ccoef_field.x_axis[:-1])/2
            y_arr = ccoef_field.y_axis[:-1] + (ccoef_field.y_axis[1:] - ccoef_field.y_axis[:-1])/2
            z_arr = ccoef_field.z_axis[:-1] + (ccoef_field.z_axis[1:] - ccoef_field.z_axis[:-1])/2

            # check midpt spacing
            # plt.figure(figsize = (15, 1))
            # grid_pts = self.ccoef_field.x_axis.detach().cpu().numpy()
            # mid_pts = x_arr.detach().cpu().numpy()
            # plt.scatter(grid_pts, np.zeros_like(grid_pts), s = 5, label = 'grid pts')
            # plt.scatter(mid_pts, np.zeros_like(mid_pts), s = 5,  label = 'mid pts')
            # plt.legend()
            # plt.show()
            # plt.close()

            # meshgrid (should be 127 by 63 by 31)
            x_mg, y_mg, z_mg = torch.meshgrid(x_arr, y_arr, z_arr, indexing = 'ij')

            lg.debug(f'x_mg shape: {x_mg.shape}')
            lg.debug(f'y_mg shape: {y_mg.shape}')
            lg.debug(f'z_mg shape: {z_mg.shape}')

            # stack, flatten
            x_xyz = torch.stack((x_mg.contiguous().view(-1), 
                                    y_mg.contiguous().view(-1), 
                                    z_mg.contiguous().view(-1)), axis = 0)
            
            # make samples on rows dim
            x_xyz = torch.transpose(x_xyz, 0, 1)

            lg.debug(f'x_xyz shape: {x_xyz.shape}')

            # compute estimated values
            x_xyz = match_field(ccoef_field.field, x_xyz)
            val_pred = ccoef_field.forward(x_xyz)

            lg.debug(f'val_pred shape: {val_pred.shape}')

            # get ground truth (code here from compute_ccoef_LUT.py)
            lg.debug(f'setting up ions in pycelp')

            import pycelp

            ion_name = 'fe_13' 
            n_levels = 50
            ion = pycelp.Ion(ion_name, nlevels=n_levels)
            
            wavelength_arr = [10747, 10798]
            line_arr = [ion.get_emissionLine(wavelength) for wavelength in wavelength_arr] # emission line objects
            num_lines = len(line_arr)

            lg.debug(f'sampling gt from CHIANTI...')
            subsample_rate = 250
            x_xyz_subsample = x_xyz.detach().cpu().numpy()[::subsample_rate]
            lg.debug(f'x_xyz_subsample size: {x_xyz_subsample.shape}')
            val_gt = np.zeros((num_lines, x_xyz_subsample.shape[0]))
            for i_line, line in enumerate(line_arr):
                for i_pt, pt in tqdm.tqdm(enumerate(x_xyz_subsample)):
                    val_gt[i_line, i_pt] = compute_ccoef(*pt, ion, line) # depends on line

            # compare
            frac_err = (val_pred[:,::subsample_rate].detach().cpu().numpy() - val_gt) / val_gt
            mean, std = np.mean(frac_err), np.std(frac_err)
            max_frac_err, min_frac_err = frac_err.max(), frac_err.min()

            lg.debug(f'mean: {mean:.6f}, std: {std:.6f}')
            lg.debug(f'max_frac_err: {max_frac_err:.6f}, min_err: {min_frac_err:.6f}')

            # plot

            # dont plot if no save
            if getattr(pipeline, "no_save", False):
                lg.debug("io.no_save = True -> skipping ccoef field test save")
                return

            plt.figure()
            hist, bin_edges = np.histogram(frac_err, bins = 100, range = (-0.25, 0.25), density = True)
            plt.stairs(hist, bin_edges)
            plt.ylabel('PDF')
            plt.xlabel('(pred - gt) / gt')
            plt.xlim(-0.25, 0.25)

            plt.tight_layout()

            subdir = dirs.figs / 'interp_test'
            subdir.mkdir(exist_ok=True, parents=True)

            fig_path = subdir / f"ccoef_midpt_test.png"
            plt.savefig(fig_path, dpi=200, bbox_inches="tight")
            lg.info(f"Saved ccoef midpt test debug figure to {fig_path}")



