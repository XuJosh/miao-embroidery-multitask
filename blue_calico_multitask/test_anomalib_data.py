from anomalib.data import Folder

dm = Folder(
    name='guizhou_embroidery',
    root='data/guizhou_embroidery_dinomaly',
    normal_dir='train/good',
    abnormal_dir='test/defect',
    normal_test_dir='test/good',
    mask_dir='ground_truth/defect',
    train_batch_size=16,
    eval_batch_size=16,
    num_workers=0,
    seed=42,
    val_split_mode='none',
)
dm.setup()
print('train', len(dm.train_data))
print('test', len(dm.test_data))
