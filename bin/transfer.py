import numpy as np
from styler import Styler
from keras import backend as K
from argparse import ArgumentParser

CONTENT_WEIGHT = 0.025
STYLE_WEIGHT = 1.0
TV_WEIGHT = 1.0
ITERATIONS = 10
CONTENT_LAYER = 'block5_conv2'
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def build_parser():
	desc = 'style transfer keras'
	parser = ArgumentParser(description=desc)

	parser.add_argument('base_img_path', metavar='base', type=str,
						help='path to base image.')
	parser.add_argument('style_img_path', metavar='style', type=str,
						help='path to style image.')
	parser.add_argument('output_img_path', metavar='output', type=str,
						help='path to output image.')
	parser.add_argument('--iters', type=int, default=ITERATIONS,
						metavar='iterations', help='Number of iterations.')
	parser.add_argument('--content_weight', type=float, default=CONTENT_WEIGHT,
						help='Weight for content feature loss')
	parser.add_argument('--style_weight', type=float, default=STYLE_WEIGHT,
						help='Weight for style feature loss')
	parser.add_argument('--tv_weight', type=float, default=TV_WEIGHT,
						help='Weight for total variation loss')
					
	return parser

def main():
	print('start of main')
	parser = build_parser()
	options = parser.parse_args()

	base_img_path = options.base_img_path
	style_img_path = options.style_img_path
	output_img_path = options.output_img_path

	iterations = options.iters
	content_weight = options.content_weight
	style_weight = options.style_weight
	total_variation_weight = options.tv_weight

	styler = Styler(base_img_path, style_img_path, output_img_path, 
					content_weight, style_weight, total_variation_weight, 
					CONTENT_LAYER, STYLE_LAYERS, iterations)

	print('making styler')
	styler.style()

if __name__ == '__main__':
	main()