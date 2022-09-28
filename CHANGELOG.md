- Easier to use background subtraction implementation, with "median" option. It is more robust against difficult tracking intervals/episodes
- Better and easier `episode` definitions with optimized parallel distribution (specially with multiple files)
- Simplified `Video.video_paths` attribute removing the old attribute `Video.video_path`
- Simplified segmentation code using the new `episode` definition
- List of blobs can reconnect after loading from .npy in almost no time

### Already existing unreleased features when this changelog was created

- Optimized 80% of the computational time of `_process_frame()` by properly removing the function `binary_fill_holes()`
- Logs more readable, with more useful information and progress bars
- Faster h5py reading implementation (by not opening and closing the h5py file for every single image, we keep them opened)
