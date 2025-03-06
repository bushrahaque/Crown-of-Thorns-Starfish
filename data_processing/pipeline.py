from convert_bbox_annotations import BBoxAnnotationConverter

for v in [0,1,2]:
        img_dir = f'tensorflow-great-barrier-reef/train_images/video_{v}/'
        label_dir = f'tensorflow-great-barrier-reef/train_images/formatted_labels_{v}/'

        converter = BBoxAnnotationConverter(
            csv_path='tensorflow-great-barrier-reef/train.csv',
            images_dir=img_dir,
            labels_dir=label_dir
        )
        converter.convert_annotations()