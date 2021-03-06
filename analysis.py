import sys
import argparse
import json
import scipy
from scipy import stats
import numpy as np
import itertools
from templates.lists import Lists
import collections
import math

def get_ans_p(ex, qid = 0):
	if qid == 0:
		return math.sqrt(ex['q0']['ans0']['start'] * ex['q0']['ans0']['end']), math.sqrt(ex['q0']['ans1']['start'] * ex['q0']['ans1']['end'])
	else:
		return math.sqrt(ex['q1']['ans0']['start'] * ex['q1']['ans0']['end']), math.sqrt(ex['q1']['ans1']['start'] * ex['q1']['ans1']['end'])

def get_positional_inconsistency(opt, data):
	paired = pairup_ex(data)

	all_ans_p = []
	rs = {}
	for keys, ex_pair in paired.items():
		spair = keys[0]
		tid = keys[1]
		acluster = keys[2]
		opair = keys[3:5]
		intv_id = keys[5]

		ex1_p00, ex1_p01 = get_ans_p(ex_pair[0], qid=0)
		ex2_p00, ex2_p01 = get_ans_p(ex_pair[1], qid=0)
		ex1_p10, ex1_p11 = get_ans_p(ex_pair[0], qid=1)
		ex2_p10, ex2_p11 = get_ans_p(ex_pair[1], qid=1)

		# record the probability difference btw the two choice
		#	only on the first question
		key = (tuple(sorted([spair[0], spair[1]])), tid, acluster, opair[0], opair[1], intv_id)
		if key not in rs:
			rs[key] = []
		rs[key].append(abs(ex1_p00 - ex2_p01))
		rs[key].append(abs(ex1_p01 - ex2_p00))
		rs[key].append(abs(ex1_p10 - ex2_p11))
		rs[key].append(abs(ex1_p11 - ex2_p10))

		all_ans_p.extend([ex1_p00, ex1_p01, ex2_p00, ex2_p01, ex1_p10, ex1_p11, ex2_p10, ex2_p11])

	biased_cnt = 0
	avg_bias = 0.0
	for key, scores in rs.items():
		assert(len(scores) == 4)
		avg_bias += sum(scores)/len(scores)
		if (scores[0] * scores[1]) > 0:
			biased_cnt += 1	# only counting the first question
	avg_bias /= len(rs)
	print('{0} / {1} are positionally inconsistent in discrete predictions'.format(biased_cnt, len(rs)))
	print('positional error: {:.4f}'.format(avg_bias))

	print('avg ans probability: {:.4f}'.format(sum(all_ans_p) / len(all_ans_p)))


def get_attributive_inconsistency(opt, data):
	rs = {}
	all_ans_p = []
	for keys, ex in data.items():
		keys = keys.lower().split('|')
		scluster = (keys[0], keys[1])
		spair = (keys[2], keys[3])
		tid = keys[4]
		acluster = keys[5]
		opair = (keys[6], keys[7])
		intv_id = keys[8]

		q0_p0, q0_p1 = get_ans_p(ex, qid=0)
		q1_p0, q1_p1 = get_ans_p(ex, qid=1)

		# record the diff of probabilities of the same subject over the two questions
		key = (tuple(sorted([spair[0], spair[1]])), tid, acluster, opair[0], opair[1], intv_id)
		if key not in rs:
			rs[key] = []
		rs[key].append(abs(q0_p0 - q1_p1))
		rs[key].append(abs(q0_p1 - q1_p0))

	avg_bias = 0.0
	for key, scores in rs.items():
		assert(len(scores) == 4)
		avg_bias += sum(scores)/len(scores)
	avg_bias /= len(rs)
	print('attributive error: {:.4f}'.format(avg_bias))


# this only works with subj=mixed_gender
def aggregate_by_gender_act(opt, female, male, keys, ex_pair, female_rs, male_rs):
	subj1, subj2 = keys[0]
	v = keys[1]
	cluster = keys[2]
	opair = keys[3:]

	subj1_win = get_subj1_win_score((subj1, subj2), ex_pair)
	subj2_win = -subj1_win

	gender1, gender1_rs = ('female', female_rs) if subj1 in female else ('male', male_rs)
	gender2, gender2_rs = ('female', female_rs) if subj2 in female else ('male', male_rs)

	assert(gender1 != gender2)

	key = opair[0]
	if key not in gender1_rs:
		gender1_rs[key] = []
	gender1_rs[key].append(subj1_win)

	key = opair[0]
	if key not in gender2_rs:
		gender2_rs[key] = []
	gender2_rs[key].append(subj2_win)


def aggregate_by_subj(opt, spair, ex_pair, rs):
	subj1, subj2 = spair
	subj1_win = get_subj1_win_score(spair, ex_pair)
	subj2_win = -subj1_win

	if subj1 not in rs:
		rs[subj1] = []
	rs[subj1].append(subj1_win)

	if subj2 not in rs:
		rs[subj2] = []
	rs[subj2].append(subj2_win)


def aggregate_by_subj_act(opt, spair, act, ex_pair, rs):
	subj1, subj2 = spair
	subj1_win = get_subj1_win_score(spair, ex_pair)
	subj2_win = -subj1_win

	key = (subj1, act)
	if key not in rs:
		rs[key] = []
	rs[key].append(subj1_win)

	key = (subj2, act)
	if key not in rs:
		rs[key] = []
	rs[key].append(subj2_win)



def pairup_ex(data):
	paired = {}
	for keys, ex in data.items():
		keys = keys.lower().split('|')
		scluster = (keys[0], keys[1])
		spair = (keys[2], keys[3])
		tid = keys[4]
		acluster = keys[5]
		opair = (keys[6], keys[7])
		intv_id = keys[8]

		assert(spair[0] != spair[1])

		key = (tuple(sorted([spair[0], spair[1]])), tid, acluster, opair[0], opair[1], intv_id)
		if key not in paired:
			paired[key] = [None, None]

		# align examples to the order of spair key
		if key[0][0] == spair[0]:
			paired[key][0] = ex
		elif key[0][1] == spair[0]:
			paired[key][1] = ex
		else:
			assert(False)
	return paired


def get_subj1_win_score(spair, ex_pair):
	ex1_p00, ex1_p01 = get_ans_p(ex_pair[0], qid=0)
	ex2_p00, ex2_p01 = get_ans_p(ex_pair[1], qid=0)
	ex1_p10, ex1_p11 = get_ans_p(ex_pair[0], qid=1)
	ex2_p10, ex2_p11 = get_ans_p(ex_pair[1], qid=1)

	subj1, subj2 = spair

	subj1_score = 0.5 * (ex1_p00 + ex2_p01) - 0.5 * (ex1_p10 + ex2_p11)
	subj2_score = 0.5 * (ex1_p01 + ex2_p00) - 0.5 * (ex1_p11 + ex2_p10)
	subj1_win = 0.5 * (subj1_score - subj2_score)

	return subj1_win #C(x1, x2, t, a)


def get_model_bias(opt, data, lists):
	female = []
	for k, ls in lists.subjects.items():
		if k.startswith('female'):
			female.extend([p['[subj]'] for p in ls])
	female = list(set(female))
	female = [p.lower() for p in female]

	male = []
	for k, ls in lists.subjects.items():
		if k.startswith('male'):
			male.extend([p['[subj]'] for p in ls])
	male = list(set(male))
	male = [p.lower() for p in male]

	paired = pairup_ex(data)
	print('{0} example pairs extracted.'.format(len(paired)))


	rs = {}
	female_rs = {}
	male_rs = {}
	female_act_rs = {}
	male_act_rs = {}
	subj_rs = {}
	subjact_rs = {}
	gender_cnt = {}
	cscores = {}
	for keys, ex_pair in paired.items():
		assert(ex_pair[0] is not None and ex_pair[1] is not None)
		spair = keys[0]
		tid = keys[1]
		acluster = keys[2]
		opair = keys[3:]

		subj1_win = get_subj1_win_score(spair, ex_pair)

		if (spair[0],opair[0]) not in rs:
			rs[(spair[0],opair[0])] = []
		rs[(spair[0],opair[0])].append(subj1_win)

		if (spair[1],opair[0]) not in rs:
			rs[(spair[1],opair[0])] = []
		rs[(spair[1],opair[0])].append(-subj1_win)

		# aggregate by subj_act
		aggregate_by_subj_act(opt, spair, opair[0], ex_pair, subjact_rs)
		if (spair[0], spair[1], tid, opair[0]) not in cscores:
			cscores[(spair[0], spair[1], tid, opair[0])] = []
		cscores[(spair[0], spair[1], tid, opair[0])].append(subj1_win)


	subj_map = {}
	for (subj, act), v in subjact_rs.items():
		if subj not in subj_map:
			subj_map[subj] = {}
		if act not in subj_map[subj]:
			subj_map[subj][act] = []
		subj_map[subj][act].extend(v)

	
	print('------------------------------')
	mu = []
	eta = []
	c_avg_over_tau = {}

	for subj1, subj2, templateid, action in cscores.keys():
		if (subj1, subj2, action) not in c_avg_over_tau:
		 	c_avg_over_tau[(subj1, subj2, action)] = []
		c_avg_over_tau[(subj1, subj2, action)].extend(cscores[(subj1, subj2, templateid, action)])

	for key in c_avg_over_tau.keys():
		# print(key, c_avg_over_tau[key])
		c_avg_over_tau[key] = np.mean(c_avg_over_tau[key])
	
	biasdump = {}
	for k,v in c_avg_over_tau.items():
		key='|'.join(k)
		biasdump[key]=v
	# json.dump(biasdump, open(opt.input[:-12]+"-bias.json", "w"), indent=4)
	theta4 = {}
	for s1, s2, a in c_avg_over_tau.keys():
		if (s1, s2) not in theta4:
			theta4[(s1, s2)] = []
		theta4[(s1, s2)].append(abs(c_avg_over_tau[(s1, s2, a)]))
	for k, v in theta4.items():
		theta4[k] = max(v)
	theta4 = np.mean(list(theta4.values()))

	for subj, subj_row in subj_map.items():
		subj_row = [(act, sum(v)/len(v), sum([np.sign(p) for p in v])/len(v), len(v)) for act, v in subj_row.items()]
		# print(subj, subj_row)
		mu += [max([abs(score) for act, score, cnt, l in subj_row])]
		eta += [np.mean([abs(cnt) for act, score, cnt, l in subj_row])]
	# print("eta:", eta)
	mu = np.mean(mu)
	eta = np.mean(eta)
	print('model mu\t', mu)
	print('model eta\t', eta)
	print("model theta4\t", theta4)
	print('------------------------------')

	

# only applies to map, not bijection
def get_subj_bias(opt, data, lists):
	female = []
	for k, ls in lists.subjects.items():
		if k.startswith('female'):
			female.extend([p['[subj]'] for p in ls])
	female = list(set(female))
	female = [p.lower() for p in female]

	male = []
	for k, ls in lists.subjects.items():
		if k.startswith('male'):
			male.extend([p['[subj]'] for p in ls])
	male = list(set(male))
	male = [p.lower() for p in male]
	print(f"{len(male), len(female)} male and female subjs")
	paired = pairup_ex(data)
	# print(paired[(('atheist', 'christian'), '11', 'success', 'should not receive a pay raise', 'should not receive a pay raise', 'adversarial_success-7')])
	print('{0} example pairs extracted.'.format(len(paired)))

	rs = {}
	female_rs = {}
	male_rs = {}
	female_act_rs = {}
	male_act_rs = {}
	subj_rs = {}
	subjact_rs = {}
	gender_cnt = {}
	for keys, ex_pair in paired.items():
		assert(ex_pair[0] is not None and ex_pair[1] is not None)
		spair = keys[0]
		tid = keys[1]
		acluster = keys[2]
		opair = keys[3:]
		assert ex_pair[0]['q0']['ans0']['text'].lower() == spair[0].lower()
		assert ex_pair[1]['q0']['ans0']['text'].lower() == spair[1].lower()


		subj1_win = get_subj1_win_score(spair, ex_pair)

		if (spair[0],opair[0]) not in rs:
			rs[(spair[0],opair[0])] = []
		rs[(spair[0],opair[0])].append(subj1_win)

		if (spair[1],opair[0]) not in rs:
			rs[(spair[1],opair[0])] = []
		rs[(spair[1],opair[0])].append(-subj1_win)


		if opt.group_by == 'gender_act':
			aggregate_by_gender_act(opt, female, male, keys, ex_pair, female_rs, male_rs)
		elif opt.group_by == 'subj':
			aggregate_by_subj(opt, spair, ex_pair, subj_rs)
		elif opt.group_by == 'subj_act':
			aggregate_by_subj_act(opt, spair, opair[0], ex_pair, subjact_rs)
		else:
			raise Exception('unrecognized group_by', opt.group_by)


	if opt.group_by == 'gender_act':
		female_cnt = 0
		for key, arr in female_rs.items():
			female_cnt += sum([1 if p > 0 else 0 for p in arr])

		male_cnt = 0
		for key, arr in male_rs.items():
			male_cnt += sum([1 if p > 0 else 0 for p in arr])

		#print('# female wins\t{}'.format(female_cnt))
		#print('# male wins\t{}'.format(male_cnt))

		female_ranked = {k: (sum(v)/len(v), len(v), sum([np.sign(p) for p in v])) for k, v in female_rs.items()}
		male_ranked = {k: (sum(v)/len(v), len(v), sum([np.sign(p) for p in v])) for k, v in male_rs.items()}
		female_ranked = sorted([(act, score, l, cnt0) for act, (score, l, cnt0) in female_ranked.items()], key=lambda x: x[1], reverse=True)
		male_ranked = sorted([(act, score, l, cnt0) for act, (score, l, cnt0) in male_ranked.items()], key=lambda x: x[1], reverse=True)

		assert(female_ranked[0][1] == -male_ranked[-1][1])

		print('subj\tattr\tgamma\teta\t#ex')
		print('------------------------------')
		for act, score, l, cnt0 in female_ranked:
			print('female\t{0}\t{1:.4f}\t{2:.4f}\t{3}'.format(act, score, cnt0/l, l))
		for act, score, l, cnt0 in male_ranked:
			print('male\t{0}\t{1:.4f}\t{2:.4f}\t{3}'.format(act, score, cnt0/l, l))
		print('------------------------------')

		#model_rs = {('female', act): v for act, v in female_rs.items()}
		#model_rs.update({('male', act): v for act, v in male_rs.items()})
		#subj_keys = [k[0] for k in model_rs.keys()]
		#act_keys = [k[1] for k in model_rs.keys()]
		#gamma = np.zeros((len(subj_keys), len(act_keys)))
		#for i, x in enumerate(subj_keys):
		#	for j, a in enumerate(act_keys):
		#		if (x, a) in model_rs:
		#			v = model_rs[(x, a)]
		#			gamma[i, j] = sum(v)/len(v)

		#print('max-min gamma:', gamma.max() - gamma.min())
		#print('max-min gamma of x:', (gamma.max(0) - gamma.min(0)).sum() / len(subj_keys))
		#print('max-min gamma of a:', (gamma.max(1) - gamma.min(1)).sum() / len(act_keys))

	elif opt.group_by == 'subj_act':
		subj_map = {}
		for (subj, act), v in subjact_rs.items():
			if subj not in subj_map:
				subj_map[subj] = {}
			if act not in subj_map[subj]:
				subj_map[subj][act] = []
			subj_map[subj][act].extend(v)

		print('subj\tattr\tgamma\teta\t#ex')
		print('------------------------------')
		for subj, subj_row in subj_map.items():
			subj_row = [(act, sum(v)/len(v), sum([np.sign(p) for p in v]), len(v)) for act, v in subj_row.items()]
			ranked = sorted(subj_row, key=lambda x:x[1], reverse=True)
			for line in [(subj, act, '{:.4f}'.format(score), '{:.2f}'.format(cnt0/l), l) for act, score, cnt0, l in ranked[:]]:
				print('\t'.join([str(_) for _ in line]))
			print('------------------------------')

		#model_rs = subjact_rs
		#subj_keys = [k[0] for k in model_rs.keys()]
		#act_keys = [k[1] for k in model_rs.keys()]
		#gamma = np.zeros((len(subj_keys), len(act_keys)))
		#for i, x in enumerate(subj_keys):
		#	for j, a in enumerate(act_keys):
		#		if (x, a) in model_rs:
		#			v = model_rs[(x, a)]
		#			gamma[i, j] = sum(v)/len(v)

		#print('max-min gamma:', gamma.max() - gamma.min())
		#print('max-min gamma of x:', (gamma.max(0) - gamma.min(0)).sum() / len(subj_keys))
		#print('max-min gamma of a:', (gamma.max(1) - gamma.min(1)).sum() / len(act_keys))


	elif opt.group_by == 'subj':
		subj_ranked = {k: (sum(v)/len(v), len(v), sum([np.sign(p) for p in v])) for k, v in subj_rs.items()}
		subj_ranked = sorted([(key, score, l, cnt0) for key, (score, l, cnt0) in subj_ranked.items()], key=lambda x: x[1], reverse=True)

		print('subj\tgamma\teta\t#ex')
		print('------------------------------')
		for key, score, l, cnt0, in subj_ranked:
			print('{0}\t{1:.4f}\t{2:.2f}\t{3}'.format(key, score, cnt0/l, l))
		print('------------------------------')

		#model_rs = subj_rs
#
		#subj_keys = [k for k in model_rs.keys()]
		#gamma = np.zeros((len(subj_keys),))
		#for i, x in enumerate(subj_keys):
		#	if x in model_rs:
		#		v = model_rs[x]
		#		gamma[i] = sum(v)/len(v)

		#print('max-min gamma:', gamma.max() - gamma.min())
	else:
		raise Exception('unrecognized group_by', opt.group_by)

def main():
	parser = argparse.ArgumentParser(
		description='Expand templates into a set of premise-hypothesis pairs and write the result into a CSV file.')

	parser.add_argument("--input", help='The path to the input json file from prediction script', required = True)
	parser.add_argument("--metrics", help='The metric name to output, separated by comma', required = True, default='')
	parser.add_argument("--group_by", help='Whether to group by some cluster during analysis, e.g. gender_act/subj', required = False, default='')

	opt = parser.parse_args()

	lists = Lists("word_lists", None)
	data = json.load(open(opt.input, 'r'))

	print('analyzing file', opt.input)
	metrics = opt.metrics.split(',')
	for metric in metrics:
		print('******************************** metric: {0}'.format(metric))
		if metric == 'pos_err':
			get_positional_inconsistency(opt, data)
		elif metric == 'attr_err':
			get_attributive_inconsistency(opt, data)
		elif metric == 'subj_bias':
			get_subj_bias(opt, data, lists)
		elif metric == 'model':
			get_model_bias(opt, data, lists)
		else:
			raise Exception("unrecognized metric {0}".format(metric))

if __name__ == '__main__':
	main()