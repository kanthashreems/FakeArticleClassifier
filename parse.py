import numpy as np
import utils
import re

parse_indices = [(2,2),(2,5),(3,3),(4,5),(5,5),(6,5),(7,5)]
def get_parse_idx_values(parse_indices, ev):
	ev = ev.split("\n")
	l = []
	for i,j in parse_indices:
		content = ev[i].split()[j]
		l.append(float(re.sub("[^.0-9]", "", content)))
	return l

def parse_file(parse_indices, textfile):
	d = utils.load(textfile)
	evals = filter(None, d.split("evallm : "))
	# print len(evals)
	parsed_vals = []
	for ev in evals:
		parsed = get_parse_idx_values(parse_indices, ev)
		parsed_vals.append(parsed)
	return parsed_vals

FNAME = "4gram/perp_wit_out_sample"
parsed_vals = parse_file(parse_indices, FNAME)
# print parsed_vals

