# BDRP
BDRP project
Colab Setup: \n
Required Installs: 
- ****
Run Deblur-NeRF: 
- Ask kou what she did to setup ****

Error logs and resolutions: 

Situation: Running Syntheic Dataset pretrained weights to record metrics
Error: subprocess.CalledProcessError: Command 'mogrify -resize 100.0% -format png *.png' returned non-zero exit status 127.
Resolution: !sudo apt-get install imagemagick 

