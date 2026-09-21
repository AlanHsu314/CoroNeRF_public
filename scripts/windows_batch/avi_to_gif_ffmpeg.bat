@echo off

for %%F in ("..\..\runs_benchmarks\multiline_ablation_ne_t\runs\fe13_si9_all4_ne_t__seed-12648430__4fa517c6\visualize\videos\microwave_orbit\*.avi") do (
    echo Processing %%F ...
    ffmpeg -y -i "%%F" -vf "fps=10,scale=640:-1:flags=lanczos,palettegen" "%%~dpnF_palette.png"
    ffmpeg -y -i "%%F" -i "%%~dpnF_palette.png" -lavfi "fps=10,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" -loop 0 "%%~dpnF_preview.gif"
    del "%%~dpnF_palette.png"
)

pause