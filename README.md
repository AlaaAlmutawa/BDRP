# BDRP Project Colab Setup

## Required Installs

<!-- Instructions or list of required installations go here -->

## Running Deblur-NeRF

<!-- Instructions for running Deblur-NeRF -->

## Additional Setup

- **Consultation with Kou**: Ask Kou about the specific setup steps she undertook for [particular aspect of the project].

## Error Logs and Resolutions

### Situation
- **Issue**: Running Synthetic Dataset pretrained weights to record metrics.
- **Error**: `subprocess.CalledProcessError`: Command 'mogrify -resize 100.0% -format png *.png' returned non-zero exit status 127.

### Resolution
- **Solution**: Install ImageMagick in the Colab environment.
- **Command**:
  ```bash
  !sudo apt-get install imagemagick


