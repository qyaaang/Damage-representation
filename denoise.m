% @Author: qyan327
% @Date:   2021-01-31 18:46:32
% @Last Modified by:   qyan327
% @Last Modified time: 2021-05-09 23:09:31


%% denoise: Wavelet transform denoise
function [xd] = demoise(x)
	% figure;
	% plot(x);
	xd = wdenoise(x, 2);
	% hold on;
	% plot(xd);
	% legend('Noisy signal','Denoised Signal')
