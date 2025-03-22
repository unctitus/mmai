#!/bin/bash

# Local destination directory
DEST="/Users/titus/Desktop/simpler_videos"
# Directory to store the converted GIFs
GIF_DIR="${DEST}/gif"
# Frame rate to use for GIF conversion (adjust if needed)
FPS=30

mkdir -p "${GIF_DIR}"

# Convert each MP4 video in the destination directory to a high-quality GIF
echo "Starting conversion of MP4 files to high-quality GIFs..."
for video in "${DEST}"/*.mp4; do
    filename=$(basename "${video}" .mp4)
    gif="${GIF_DIR}/${filename}.gif"
    palette="/tmp/${filename}_palette.png"
    
    echo "Generating palette for ${video}"
    ffmpeg -y -i "${video}" -vf "fps=${FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2,palettegen" "${palette}"
    
    echo "Converting ${video} to ${gif} with high quality"
    ffmpeg -i "${video}" -i "${palette}" -filter_complex "fps=${FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,paletteuse" "${gif}"
    
    # Remove temporary palette file
    rm "${palette}"
done

echo "Conversion complete. All GIFs are saved in ${GIF_DIR}"
