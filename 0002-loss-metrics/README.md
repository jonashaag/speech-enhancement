Problem: PESQ is well studied speech quality metric, good for reverb scoring, but reference implementation is CPU only and no GPU implementation available.

Maybe PMSQE works better? http://sigmat.ugr.es/PMSQE/

Also look into wMSE.

Other metrics that work ok are SI-SDR (but does not work well with phase information) and wSDR (SDR adapted for phase information). Or simply spectrogram MSE. (Wave MSE does not work well at all.)

Google Visqol metric is good but slower than PESQ.

POLQA is commercial with no free reference implementation

Or simply use DCT based spectrograms to avoid phase problem? https://arxiv.org/pdf/1910.07840.pdf

PMSQE seems to not perform well: On Loss Functions for Supervised Monaural Time-Domain Speech Enhancement

SI-SDR seems to work fine for most cases, but should account for time-shifted signals when evaluating systems that were trained using a spectrogram loss.

https://ieeexplore.ieee.org/abstract/document/9052949

https://github.com/gabrielmittag/NISQA

https://ieeexplore.ieee.org/abstract/document/7602933
