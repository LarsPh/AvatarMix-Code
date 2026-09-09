import os
import glob
import subprocess
import tempfile


def make_video(out_dir, video_name_base, frame_pattern="%04d.jpg", frame_rate=30):

    image_files = sorted(glob.glob(os.path.join(out_dir, frame_pattern)))
    if not image_files:
        print(f"No images found in {out_dir} matching pattern {frame_pattern}. Video not created.")
        return


    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        for img_path in image_files:


            tmp_file.write(f"file '{os.path.abspath(img_path)}'\n")
        temp_file_list_path = tmp_file.name

    try:
        video_path = os.path.join(out_dir, f'{video_name_base}.mp4')


        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_file_list_path,
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-r', str(frame_rate),
            video_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video generated successfully at: {video_path}")
    except Exception as e:
        print(f"Failed to generate video: {str(e)}")
    finally:

        if os.path.exists(temp_file_list_path):
            os.remove(temp_file_list_path)
