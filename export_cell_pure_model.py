if __name__ == "__main__":
    import torch
    model_path = "/media/zhouzhihao/DeepInformatic_dataset/lihansheng/bxr/code/bxr_wo_consist_low_memo/checkpoint/bxr_20240104_resnet50_eos_coef_0_5_beiertong.pth"
    save_path = "/media/zhouzhihao/DeepInformatic_dataset/lihansheng/bxr/code/bxr_wo_consist_low_memo/checkpoint/bxr_beiertong_pure_20240104.pth"
    checkpoint_all = torch.load(model_path)
    checkpoint = checkpoint_all['model']
    torch.save(checkpoint, save_path)