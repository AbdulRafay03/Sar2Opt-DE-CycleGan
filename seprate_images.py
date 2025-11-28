import os
import shutil

src = r"C:\Users\Abdul Rafay\Desktop\Sar2Opt-DE-CycleGAN\Sar2Opt-DE-CycleGan\results\train1_cyclegan\test_latest\images - Copy"      # folder containing all files
dst = r"results\train1_cyclegan\test_latest\seprated"           # folder where separated folders will be created

categories = {
    "fake_A": "fakeA",
    "fake_B": "fakeB",
    "real_A": "realA",
    "real_B": "realB",
    "rec_A":  "recA",
    "rec_B":  "recB"
}

# create output folders
for c in categories.values():
    os.makedirs(os.path.join(dst, c), exist_ok=True)

# move files based on suffix
for filename in os.listdir(src):
    for key, folder in categories.items():
        if filename.endswith(f"{key}.png") or filename.endswith(f"{key}.jpg"):
            shutil.move(
                os.path.join(src, filename),
                os.path.join(dst, folder, filename)
            )
            break
