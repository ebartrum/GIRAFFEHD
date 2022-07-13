import os
import sys
import argparse
import torch
from torchvision import utils
from model import GIRAFFEHDGenerator
from tqdm import tqdm
import math
import torch.nn.functional as F

def get_interval(args):
    if args.control_i == 4:
        if args.range_u is None:
            p0 = torch.tensor([[args.ckpt_args.range_u[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.ckpt_args.range_u[1]]]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([[args.range_u[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.range_u[1]]]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 5:
        if args.range_v is None:
            p0 = torch.tensor([[args.ckpt_args.range_v[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.ckpt_args.range_v[1]]]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([[args.range_v[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.range_v[1]]]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 6:
        print('Changing radius not implemented')
        sys.exit()

    elif args.control_i == 7:
        if args.scale_range_min is None:
            p0 = torch.tensor([args.ckpt_args.scale_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.ckpt_args.scale_range_max]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([args.scale_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.scale_range_max]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 8:
        if args.translation_range_min is None:
            p0 = torch.tensor([args.ckpt_args.translation_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.ckpt_args.translation_range_max]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([args.translation_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.translation_range_max]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 9:
        if args.rotation_range is None:
            p0 = torch.tensor([[args.ckpt_args.rotation_range[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.ckpt_args.rotation_range[1]]]).repeat(
                args.batch, 1).to(args.device)

            rotation_radians = 0.45
            rotation_proportion = rotation_radians / (2*math.pi) # unit here seems to be radians / 2pi
            p0 = -torch.ones_like(p0)*rotation_proportion + 0.5 # 0.5 is frontal pose (pi radians)
            p1 = torch.ones_like(p1)*rotation_proportion + 0.5
        else:
            p0 = torch.tensor([[args.rotation_range[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.rotation_range[1]]]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 10:
        if args.bg_translation_range_min is None:
            p0 = torch.tensor([args.ckpt_args.bg_translation_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.ckpt_args.bg_translation_range_max]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([args.bg_translation_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.bg_translation_range_max]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 11:
        if args.bg_rotation_range is None:
            p0 = torch.tensor([[args.ckpt_args.bg_rotation_range[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.ckpt_args.bg_rotation_range[1]]]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([[args.bg_rotation_range[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.bg_rotation_range[1]]]).repeat(
                args.batch, 1).to(args.device)

    return p0, p1


def eval(args, generator, max_resolution=256):
    if args.control_i == 0:
        category_dir = "shape"
    elif args.control_i == 1:
        category_dir = "fg"
    elif args.control_i == 2:
        category_dir = "bg_shape"
    elif args.control_i == 3:
        category_dir = "bg_appearance"
    elif args.control_i == 5:
        category_dir = "elevation"
    elif args.control_i == 9:
        category_dir = "azimuth"
    else:
        raise ValueError('Unknown category_dir')

    generator.eval()

    num_objs_processed = 0

    if args.control_i in list(range(0,4)):
        while num_objs_processed < args.num_objs:
            print(f"Current num_objs_processed: {num_objs_processed}")
            img_rep = generator.get_rand_rep(args.batch)

            #set defaults to 0.5
            # img_rep[4] = 0.5*torch.ones_like(img_rep[4])
            img_rep[5] = 0.4584*torch.ones_like(img_rep[5])
            # img_rep[7] = 0.5*torch.ones_like(img_rep[7])
            # img_rep[8] = torch.zeros_like(img_rep[8])
            img_rep[9] = 0.5*torch.ones_like(img_rep[9])
            # img_rep[10] = 0.5*torch.ones_like(img_rep[10])
            # img_rep[11] = 0.5*torch.ones_like(img_rep[11])

            for i in tqdm(range(args.n_sample)):
                _img_rep = generator.get_rand_rep(args.batch)
                img_rep[args.control_i] = _img_rep[args.control_i]
                out_li = generator(img_rep=img_rep, inject_index=args.inj_idx, mode='eval', return_ids=[0, 6])
                img_batch = out_li[0]
                depth_map = out_li[-1]
                img_batch = F.interpolate(img_batch, max_resolution)

                for img_id, img in enumerate(img_batch):
                    outdir = os.path.join("eval", category_dir, f"obj_{img_id + num_objs_processed}")
                    os.makedirs(outdir, exist_ok=True)
                    filepath = os.path.join(outdir, f"{i}.png")
                    utils.save_image(
                        img,
                        filepath,
                        normalize=True,
                        range=(-1, 1),
                    )
                    depth_filepath = os.path.join(outdir, f"depth_{i}.png")
                    depth_img = depth_map[[img_id]]
                    depth_img = F.interpolate(depth_img.unsqueeze(0), 128).squeeze(0)
                    depth_img = (depth_img+1)/2
                    utils.save_image(
                        depth_img,
                        depth_filepath,
                        normalize=False,
                    )

            num_objs_processed += args.batch

    if args.control_i in list(range(4,12)):
        p0, p1 = get_interval(args)
        delta = (p1 - p0) / (args.n_sample - 1)
        with torch.no_grad():
            while num_objs_processed < args.num_objs:
                print(f"Current num_objs_processed: {num_objs_processed}")
                img_rep = generator.get_rand_rep(args.batch)

                #set defaults to 0.5
                # img_rep[4] = 0.5*torch.ones_like(img_rep[4])
                img_rep[5] = 0.4584*torch.ones_like(img_rep[5])
                # img_rep[7] = torch.ones_like(img_rep[7])

                # img_rep[9] = 0.5*torch.ones_like(img_rep[9])
                # img_rep[10] = 0.5*torch.ones_like(img_rep[10])
                # img_rep[11] = 0.5*torch.ones_like(img_rep[11])

                for i in tqdm(range(args.n_sample)):
                    p = p0 + delta * i
                    img_rep[args.control_i] = p
                    out_li = generator(img_rep=img_rep, inject_index=args.inj_idx, mode='eval', return_ids=[0, 5, 6], depth_res=64)
                    img_batch = out_li[0]
                    depth_map = out_li[-1]
                    alpha_map = out_li[-2]
                    alpha_map = (alpha_map>0.1).to(torch.uint8)

                    img_batch = F.interpolate(img_batch, max_resolution)
                    alpha_map = F.interpolate(alpha_map, max_resolution)

                    for img_id, img in enumerate(img_batch):
                        outdir = os.path.join("eval", category_dir, f"obj_{img_id + num_objs_processed}")
                        os.makedirs(outdir, exist_ok=True)
                        filepath = os.path.join(outdir, f"{i}.png")
                        utils.save_image(
                            img,
                            filepath,
                            normalize=True,
                            range=(-1, 1),
                        )

                        depth_filepath = os.path.join(outdir, f"depth_{i}.pt")
                        torch.save(depth_map[img_id], depth_filepath)
                        alpha_filepath = os.path.join(outdir, f"alpha_{i}.pt")
                        torch.save(alpha_map[img_id], alpha_filepath)

                        #save depth img
                        depth_img_filepath = os.path.join(outdir, f"depth_{i}.png")
                        depth_img = depth_map[[img_id]]
                        depth_img = F.interpolate(depth_img.unsqueeze(0), 128).squeeze(0)
                        depth_img = (depth_img+1)/2
                        utils.save_image(
                            depth_img,
                            depth_img_filepath,
                            normalize=False,
                        )

                        #save alpha img
                        alpha_img_filepath = os.path.join(outdir, f"alpha_{i}.png")
                        alpha_img = alpha_map[[img_id]].float()
                        utils.save_image(
                            alpha_img,
                            alpha_img_filepath,
                            normalize=False,
                        )


                num_objs_processed += args.batch

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Giraffe trainer")

    parser.add_argument('--ckpt', type=str, default=None, help='path to the checkpoint')

    parser.add_argument('--batch', type=int, default=16, help='batch size')

    parser.add_argument('--num_objs', type=int, default=16, help='Number of objs')

    parser.add_argument('--n_sample', type=int, default=8, help='number of the samples generated')
    parser.add_argument('--inj_idx', type=int, default=-1, help='inject index for evaluation')

    parser.add_argument("--control_i", type=int, default=0, help='control index')
    # 0: fg_shape; 1: fg_app; 2: bg_shape;    3: bg_app;   4: camera rotation angle;  5: elevation angle;
    # --: radius;  7: scale;  8: translation; 9: rotation; 10: bg translation;        11: bg rotation;

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.device = device

    assert args.ckpt is not None
    print("load model:", args.ckpt)

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    args.ckpt_args = ckpt['args']

    # change interpolation ranges if needed
    args.scale_range_min = None  # [0.2, 0.16, 0.16]
    args.scale_range_max = None  # [0.25, 0.2, 0.2]

    args.translation_range_min = None  # [-0.22, -0.12, -0.06]
    args.translation_range_max = None  # [0.22, 0.12, 0.08]

    args.rotation_range = None  # [0., 1.]

    args.bg_translation_range_min = None  # [-0.2, -0.2, 0.]
    args.bg_translation_range_max = None  # [0.2, 0.2, 0.]

    args.bg_rotation_range = None  # [0., 0.]

    args.range_u = None  # [0., 0.]

    args.range_v = None  # [0.41667, 0.5]


    if args.inj_idx == -1:
        if args.ckpt_args.size == 256:
            args.inj_idx = 2
        elif args.ckpt_args.size == 512:
            args.inj_idx = 4
        elif args.ckpt_args.size == 1024:
            args.inj_idx = 4

    generator = GIRAFFEHDGenerator(
        device=device,
        z_dim=args.ckpt_args.z_dim,
        z_dim_bg=args.ckpt_args.z_dim_bg,
        size=args.ckpt_args.size,
        resolution_vol=args.ckpt_args.res_vol,
        feat_dim=args.ckpt_args.feat_dim,
        range_u=args.ckpt_args.range_u,
        range_v=args.ckpt_args.range_v,
        fov=args.ckpt_args.fov,
        scale_range_max=args.ckpt_args.scale_range_max,
        scale_range_min=args.ckpt_args.scale_range_min,
        translation_range_max=args.ckpt_args.translation_range_max,
        translation_range_min=args.ckpt_args.translation_range_min,
        rotation_range=args.ckpt_args.rotation_range,
        bg_translation_range_max=args.ckpt_args.bg_translation_range_max,
        bg_translation_range_min=args.ckpt_args.bg_translation_range_min,
        bg_rotation_range=args.ckpt_args.bg_rotation_range,
        refine_n_styledconv=2,
        refine_kernal_size=3,
        grf_use_mlp=args.ckpt_args.grf_use_mlp,
        pos_share=args.ckpt_args.pos_share,
        use_viewdirs=args.ckpt_args.use_viewdirs,
        grf_use_z_app=args.ckpt_args.grf_use_z_app,
        fg_gen_mask=args.ckpt_args.fg_gen_mask
    ).to(device)

    generator.load_state_dict(ckpt["g_ema"])

    with torch.no_grad():
        eval(args, generator)
