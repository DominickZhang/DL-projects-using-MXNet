from mxnet import gluon
from mxnet import nd
from mxnet.contrib import text

#Create dictionary
dic = text.embedding.create('glove', pretrained_file_name = 'glove.6B.50d.txt')
print(len(dic))

#Define simularity function
def cos_sim(x, y):
	return nd.dot(x, y)/((nd.norm(x)*nd.norm(y)))

#define function of finding nearest neighbors
def norm_vecs_by_row(x):
	return x/nd.sqrt(nd.sum(x*x, axis=1)).reshape((-1, 1))

def get_knn(token_embedding, k, word):
	word_vec = token_embedding.get_vecs_by_tokens([word]).reshape((-1, 1))
	vocab_vecs = norm_vecs_by_row(token_embedding.idx_to_vec)
	dot_prod = nd.dot(vocab_vecs, word_vec)
	indices = nd.topk(dot_prod.reshape((len(token_embedding),)), k=k+2, ret_typ='indices')
	indices = [int(i.asscalar()) for i in indices]
	return token_embedding.to_tokens(indices[2:])

print(get_knn(dic, 5, 'baby'))
print(cos_sim(dic.get_vecs_by_tokens('baby'), dic.get_vecs_by_tokens('babies')))
print(cos_sim(dic.get_vecs_by_tokens('baby'), dic.get_vecs_by_tokens('boy')))

def get_top_k_by_analogy(token_embedding, k, word1, word2, word3):
	word_vecs = token_embedding.get_vecs_by_tokens([word1, word2, word3])
	word_diff = (word_vecs[1] - word_vecs[0] + word_vecs[2]).reshape((-1, 1))
	vocab_vecs = norm_vecs_by_row(token_embedding.idx_to_vec)
	dot_prod = nd.dot(vocab_vecs, word_diff)
	indices = nd.topk(dot_prod.reshape((len(token_embedding), )), k=k+1, ret_typ='indices')
	indices = [int(i.asscalar()) for i in indices]

	if token_embedding.to_tokens(indices[0]) == token_embedding.unknown_token:
		return token_embedding.to_tokens(indices[1:])
	else:
		return token_embedding.to_tokens(indices[:-1])

analwords = get_top_k_by_analogy(dic, 5, 'man', 'woman', 'son')
print(analwords)

def cos_sim_word_analogy(token_embedding, word1, word2, word3, word4):
	words = [word1, word2, word3, word4]
	vecs = token_embedding.get_vecs_by_tokens(words)
	return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])

for iword in analwords:
	print(cos_sim_word_analogy(dic, 'man', 'woman', 'son', iword))

analwords1 = get_top_k_by_analogy(dic, 5, 'beijing', 'china', 'tokyo')
print(analwords1)
for iword in analwords1:
	print(cos_sim_word_analogy(dic, 'beijing', 'china', 'tokyo', iword))

analwords2 = get_top_k_by_analogy(dic, 5, 'bad', 'worst', 'big')
print(analwords2)
for iword in analwords2:
	print(cos_sim_word_analogy(dic, 'bad', 'worst', 'big', iword))
