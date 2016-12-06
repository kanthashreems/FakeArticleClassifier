import numpy as np
import utils
import re
parse_indices_1gram = [(2,2),(2,5),(3,3),(4,5)]
parse_indices_1gram_inclusive = [(3,2),(2,5),(3,3),(4,5)]
parse_indices_2gram = [(2,2),(2,5),(3,3),(4,5),(5,5)]
parse_indices_2gram_inclusive  = [(3,2),(2,5),(3,3),(4,5),(5,5)]
parse_indices_3gram = [(2,2),(2,5),(3,3),(4,5),(5,5),(6,5)]
parse_indices_3gram_inclusive = [(3,2),(2,5),(3,3),(4,5),(5,5),(6,5)]
parse_indices_4gram = [(2,2),(2,5),(3,3),(4,5),(5,5),(6,5),(7,5)]
parse_indices_4gram_inclusive = [(3,2),(2,5),(3,3),(4,5),(5,5),(6,5),(7,5)]
parse_indices_5gram = [(2,2),(2,5),(3,3),(4,5),(5,5),(6,5),(7,5),(8,5)]
parse_indices_5gram_inclusive = [(3,2),(2,5),(3,3),(4,5),(5,5),(6,5),(7,5),(8,5)]
def get_parse_idx_values(parse_indices, ev):
	ev = ev.split("\n")
	l = []
	for i,j in parse_indices[:1]:
		content = ev[i].split()[j]
		# print content
		l.append(float(re.sub("[^.0-9]", "", content)))
	return l

def parse_file(parse_indices, textfile):
	d = utils.load(textfile)
	evals = filter(None, d.split("evallm : "))
	print textfile
	# print len(evals)
	parsed_vals = []
	for ev in evals:
		parsed = get_parse_idx_values(parse_indices, ev)
		parsed_vals.append(parsed)
	return parsed_vals

FNAME = "4gram/perp_wit_out_sample"
parsed_vals = parse_file(parse_indices_3gram, FNAME)
print parsed_vals

