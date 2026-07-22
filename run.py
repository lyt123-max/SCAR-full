from __future__ import annotations

import argparse

from coremad import CoReMADConfig, CoReMADTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoReM-AD v2.0")
    parser.add_argument("--stage", type=str, default="full", choices=["stage_a", "stage_b", "test", "full"])
    parser.add_argument("--dataset", type=str, default="MSL")
    parser.add_argument("--data_root", type=str, default="./dataset/anomaly_detect")
    parser.add_argument("--artifact_root", type=str, default="./artifacts")
    parser.add_argument("--experiment_name", type=str, default="coremad_msl")
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--memory_batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--memory_build_stride", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--val_gap", type=int, default=1)
    parser.add_argument("--val_min_train_windows", type=int, default=50)
    parser.add_argument(
        "--val_split_mode",
        type=str,
        default=None,
        choices=["tail", "interleaved"],
        help="Validation split mode. Defaults to interleaved for PSM and tail for other datasets.",
    )
    parser.add_argument("--stage_a_epochs", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--scheduler_eta_min_ratio", type=float, default=0.01)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_sizes", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--d_z", type=int, default=128)
    parser.add_argument("--top_M", type=int, default=50)
    parser.add_argument("--top_K", type=int, default=20)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--clean_ratio", type=float, default=0.02)
    parser.add_argument("--state_prototype_count", type=int, default=0)
    parser.add_argument("--prototype_top_p", type=int, default=3)
    parser.add_argument("--prototype_candidate_cap", type=int, default=512)
    parser.add_argument("--soft_candidate_tau", type=float, default=1.0)
    parser.add_argument("--support_score_tau", type=float, default=1.0)
    parser.add_argument("--support_score_eps", type=float, default=1e-8)
    parser.add_argument("--include_soft_support_in_fusion", type=int, default=0)
    parser.add_argument("--use_prototype_support", type=int, default=1)
    parser.add_argument("--use_faiss", type=int, default=1)
    parser.add_argument("--faiss_use_gpu", type=int, default=1)
    parser.add_argument("--faiss_exact_threshold", type=int, default=100000)
    parser.add_argument("--faiss_ivf_nprobe", type=int, default=16)
    parser.add_argument("--coreset_keep_ratio", type=float, default=1.0)
    parser.add_argument("--coreset_max_patches_per_scale", type=int, default=200000)
    parser.add_argument("--coreset_fps_threshold", type=int, default=100000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--max_test_windows", type=int, default=0)
    parser.add_argument(
        "--sequence_score_aggregation",
        type=str,
        default="p95",
        choices=["p95", "top5_mean", "max"],
    )
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--n_mask_groups", type=int, default=4)
    parser.add_argument("--completion_dropout", type=float, default=0.1)
    parser.add_argument("--lambda_pred", type=float, default=0.5)
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--use_stsd_decomposition", type=int, default=1)
    parser.add_argument("--use_channel_modulation", type=int, default=1)
    parser.add_argument("--use_two_level_retrieval", type=int, default=1)
    parser.add_argument("--use_context_key_retrieval", type=int, default=1)
    parser.add_argument("--use_completion_head", type=int, default=1)
    parser.add_argument("--use_completion_self_cleaning", type=int, default=1)
    parser.add_argument("--use_completion_score_fusion", type=int, default=1)
    parser.add_argument(
        "--evaluation_score_key",
        type=str,
        default="cdf_mean",
        choices=["cdf_max", "cdf_mean", "cdf_mean_soft_support", "cdf_softmax", "raw_max", "zscore_mean"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = str(args.dataset).upper()
    val_split_mode = args.val_split_mode or ("interleaved" if dataset_name == "PSM" else "tail")
    if args.val_split_mode is None:
        print(f"[run.py] auto val_split_mode={val_split_mode} for dataset={dataset_name}")
    train_batch_size = args.train_batch_size or args.batch_size
    val_batch_size = args.val_batch_size or args.batch_size
    memory_batch_size = args.memory_batch_size or args.batch_size
    test_batch_size = args.test_batch_size or args.batch_size
    config = CoReMADConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        experiment_name=args.experiment_name,
        resume=bool(args.resume),
        seq_len=args.seq_len,
        batch_size=train_batch_size,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        memory_batch_size=memory_batch_size,
        test_batch_size=test_batch_size,
        train_stride=args.train_stride,
        test_stride=args.test_stride,
        memory_build_stride=args.memory_build_stride,
        val_ratio=args.val_ratio,
        val_gap=bool(args.val_gap),
        val_min_train_windows=args.val_min_train_windows,
        val_split_mode=val_split_mode,
        stage_a_epochs=args.stage_a_epochs,
        early_stop_patience=args.early_stop_patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        scheduler_eta_min_ratio=args.scheduler_eta_min_ratio,
        mask_ratio=args.mask_ratio,
        n_mask_groups=args.n_mask_groups,
        completion_dropout=args.completion_dropout,
        lambda_pred=args.lambda_pred,
        lambda_smooth=args.lambda_smooth,
        use_stsd_decomposition=bool(args.use_stsd_decomposition),
        use_channel_modulation=bool(args.use_channel_modulation),
        use_two_level_retrieval=bool(args.use_two_level_retrieval),
        use_context_key_retrieval=bool(args.use_context_key_retrieval),
        use_completion_head=bool(args.use_completion_head),
        use_completion_self_cleaning=bool(args.use_completion_self_cleaning),
        use_completion_score_fusion=bool(args.use_completion_score_fusion),
        evaluation_score_key=args.evaluation_score_key,
        patch_sizes=args.patch_sizes,
        d_z=args.d_z,
        top_M=args.top_M,
        top_K=args.top_K,
        knn_k=args.knn_k,
        clean_ratio=args.clean_ratio,
        state_prototype_count=args.state_prototype_count,
        prototype_top_p=args.prototype_top_p,
        prototype_candidate_cap=args.prototype_candidate_cap,
        soft_candidate_tau=args.soft_candidate_tau,
        support_score_tau=args.support_score_tau,
        support_score_eps=args.support_score_eps,
        include_soft_support_in_fusion=bool(args.include_soft_support_in_fusion),
        use_prototype_support=bool(args.use_prototype_support),
        use_faiss=bool(args.use_faiss),
        faiss_use_gpu=bool(args.faiss_use_gpu),
        faiss_exact_threshold=args.faiss_exact_threshold,
        faiss_ivf_nprobe=args.faiss_ivf_nprobe,
        coreset_keep_ratio=args.coreset_keep_ratio,
        coreset_max_patches_per_scale=args.coreset_max_patches_per_scale,
        coreset_fps_threshold=args.coreset_fps_threshold,
        num_workers=args.num_workers,
        max_train_windows=args.max_train_windows,
        max_test_windows=args.max_test_windows,
        sequence_score_aggregation=args.sequence_score_aggregation,
        device=args.device or CoReMADConfig().device,
        seed=args.seed,
    )
    trainer = CoReMADTrainer(config)

    if args.stage == "stage_a":
        trainer.run_stage_a()
    elif args.stage == "stage_b":
        trainer.run_stage_b()
    elif args.stage == "test":
        trainer.run_test()
    else:
        trainer.run_full()


if __name__ == "__main__":
    main()
