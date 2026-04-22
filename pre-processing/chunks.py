from pathlib import Path
import shutil

CHUNK_SIZE = 10000
BASE = Path(r"C:\Users\fastf\Downloads\rclone-v1.73.2-windows-amd64\rclone-v1.73.2-windows-amd64\genimage\SD1_4\sd1_4\data")  # change this

def split_to_imagefolder_chunks(split_dir: Path, output_dir: Path, classes=("fake", "real"), chunk_size=10000, move_files=False):
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_files = {}
    max_chunks = 0

    # Collect files for each class
    for cls in classes:
        class_dir = split_dir / cls
        files = sorted([p for p in class_dir.iterdir() if p.is_file()])
        class_files[cls] = files
        num_chunks = (len(files) + chunk_size - 1) // chunk_size
        max_chunks = max(max_chunks, num_chunks)

    # Create chunks and place files
    for cls in classes:
        files = class_files[cls]

        for i, file_path in enumerate(files):
            chunk_idx = i // chunk_size
            chunk_class_dir = output_dir / f"chunk_{chunk_idx}" / cls
            chunk_class_dir.mkdir(parents=True, exist_ok=True)

            destination = chunk_class_dir / file_path.name

            if move_files:
                shutil.move(str(file_path), str(destination))
            else:
                shutil.copy2(str(file_path), str(destination))

        # Make sure empty class folders exist in every chunk
        # so all chunks keep the same class structure
        for chunk_idx in range(max_chunks):
            (output_dir / f"chunk_{chunk_idx}" / cls).mkdir(parents=True, exist_ok=True)

    print(f"Done: {split_dir} -> {output_dir}")
    for cls in classes:
        print(f"  {cls}: {len(class_files[cls])} files")
    print(f"  Chunks created: {max_chunks}")


# TRAIN
split_to_imagefolder_chunks(
    split_dir=BASE / "train",
    output_dir=BASE / "train_chunks",
    classes=("fake", "real"),
    chunk_size=CHUNK_SIZE,
    move_files=True,   # set True to move instead of copy
)

# VAL
split_to_imagefolder_chunks(
    split_dir=BASE / "val",
    output_dir=BASE / "val_chunks",
    classes=("fake", "real"),
    chunk_size=CHUNK_SIZE,
    move_files=True,   # set True to move instead of copy
)


#TEST
split_to_imagefolder_chunks(
    split_dir=BASE / "test",
    output_dir=BASE / "test_chunks",
    classes=("fake", "real"),
    chunk_size=CHUNK_SIZE,
    move_files=True,   # set True to move instead of copy
)