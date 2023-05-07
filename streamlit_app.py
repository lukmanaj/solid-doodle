import streamlit as st
import subprocess
import argparse
import collections
import sys
import os
import numpy as np
import pandas as pd
import tempfile




BATCH_SIZE = 500

try:
    import cupy
except ImportError:
    cupy = None


def supports_cupy():
    return cupy is not None


def get_cupy():
    return cupy


def get_array_module(x):
    if cupy is not None:
        return cupy.get_array_module(x)
    else:
        return np


def asnumpy(x):
    if cupy is not None:
        return cupy.asnumpy(x)
    else:
        return np.asarray(x)
 
 
def read(file, threshold=0, vocabulary=None, dtype='float'):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))


def write(words, matrix, file):
    m = asnumpy(matrix)
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)


def length_normalize(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, xp.newaxis]


def mean_center(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    matrix -= avg


def length_normalize_dimensionwise(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms


def mean_center_embeddingwise(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=1)
    matrix -= avg[:, xp.newaxis]


def normalize(matrix, actions):
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            mean_center(matrix)
        elif action == 'unitdim':
            length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            mean_center_embeddingwise(matrix) 
 
def convert_to_np(lst, dtype='float'):
  count = len(lst)
  dim = len(lst[0])

  matrix = np.empty((count, dim), dtype=dtype)
  for i in range(count):
    matrix[i] = np.asarray(lst[i], dtype=dtype)

  return matrix

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
  xp = get_array_module(m)
  n = m.shape[0]
  ans = xp.zeros(n, dtype=m.dtype)
  if k <= 0:
    return ans
  if not inplace:
    m = xp.array(m)
  ind0 = xp.arange(n)
  ind1 = xp.empty(n, dtype=int)
  minimum = m.min()
  for i in range(k):
    m.argmax(axis=1, out=ind1)
    ans += m[ind0, ind1]
    m[ind0, ind1] = minimum
  return ans / k


def main():
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Select candidate translations giving sentences in two languages')
  parser.add_argument('-k', '--cohere_api_key', required=True, type=str, help='your personal cohere api key')
  parser.add_argument('-s', '--src_sentences', default=sys.stdin.fileno(), help='the file containing source sentences.')
  parser.add_argument('-t', '--trg_sentences', default=sys.stdin.fileno(), help='the file containing target sentences.')
  parser.add_argument('-m', '--model', required=True, type=str, help='cohere multilingual model name.')
  parser.add_argument('-o', '--output', default='', help='path to save the translations.')
  parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
  parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
  parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
  parser.add_argument('-n', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
  parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
  parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
  parser.add_argument('--seed', type=int, default=0, help='the random seed')
  parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
  parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
  args = parser.parse_args()

  # Choose the right dtype for the desired precision
  if args.precision == 'fp16':
    dtype = 'float16'
  elif args.precision == 'fp32':
    dtype = 'float32'
  elif args.precision == 'fp64':
    dtype = 'float64'

  if not os.path.isdir(args.output):
    os.makedirs(args.output)
    print('creating output directory: done')

  # Initialise cohere embedding
  api_key = args.cohere_api_key
  co = cohere.Client(f"{api_key}")

  # Get source embeddings
  with open(args.src_sentences, 'r') as f:  
    src_sents = f.readlines()
    src_sents = [line.strip() for line in src_sents]

  response = co.embed(texts=src_sents, model=args.model)  
  x = response.embeddings
  x = convert_to_np(x)

  # Get target embeddings
  with open(args.trg_sentences, 'r') as f:  
    trg_sents = f.readlines()
    trg_sents = [line.strip() for line in trg_sents]
  
  response = co.embed(texts=trg_sents, model=args.model)  
  z = response.embeddings
  z = convert_to_np(z)

  # NumPy/CuPy management
  if args.cuda:
    if not supports_cupy():
      print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
      sys.exit(-1)
    xp = get_cupy()
    src_embeddings = xp.asarray(x)
    trg_embeddings = xp.asarray(z)
  else:
    print('cuda not provided, using cpu.')
    xp = np
  xp.random.seed(args.seed)

  # Length normalize embeddings so their dot product effectively computes the cosine similarity
  if not args.dot:
    embeddings.length_normalize(x)
    embeddings.length_normalize(z)

    print('normarlize embeddings: done')

  # Build sent to index map
  src_sent2ind = {sent: i for i, sent in enumerate(src_sents)}
  print('build source sent to index map: done')
  print('length of source embedding', len(src_sent2ind))
  
  trg_sent2ind = {sent: i for i, sent in enumerate(trg_sents)}
  print('build target word to index map: done')
  print('length of target embedding', len(trg_sent2ind))

  src = [ind for ind in src_sent2ind.values()]

  # Find translations
  translation = collections.defaultdict(int)

  # Standard nearest neighbor
  if args.retrieval == 'nn':
    for i in range(0, len(src), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(src))
      similarities = x[src[i:j]].dot(z.T)
      nn = similarities.argmax(axis=1).tolist()
      for k in range(j-i):
        translation[src[i+k]] = nn[k]
  
  # Inverted nearest neighbor
  elif args.retrieval == 'invnn':
    best_rank = np.full(len(src), x.shape[0], dtype=int)
    best_sim = np.full(len(src), -100, dtype=dtype)
    for i in range(0, z.shape[0], BATCH_SIZE):
      j = min(i + BATCH_SIZE, z.shape[0])
      similarities = z[i:j].dot(x.T)
      ind = (-similarities).argsort(axis=1)
      ranks = asnumpy(ind.argsort(axis=1)[:, src])
      sims = asnumpy(similarities[:, src])
      for k in range(i, j):
        for l in range(len(src)):
          rank = ranks[k-i, l]
          sim = sims[k-i, l]
          if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
            best_rank[l] = rank
            best_sim[l] = sim
            translation[src[l]] = k
  
  # Inverted softmax
  elif args.retrieval == 'invsoftmax':
    sample = xp.arange(x.shape[0]) if args.inv_sample is None else xp.random.randint(0, x.shape[0], args.inv_sample)
    partition = xp.zeros(z.shape[0])
    for i in range(0, len(sample), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(sample))
      partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
    for i in range(0, len(src), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(src))
      p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
      nn = p.argmax(axis=1).tolist()
      for k in range(j-i):
        translation[src[i+k]] = nn[k]
  
  # Cross-domain similarity local scaling
  elif args.retrieval == 'csls':
    knn_sim_bwd = xp.zeros(z.shape[0])
    for i in range(0, z.shape[0], BATCH_SIZE):
      j = min(i + BATCH_SIZE, z.shape[0])
      knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
    for i in range(0, len(src), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(src))
      similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
      nn = similarities.argmax(axis=1).tolist()
      for k in range(j-i):
        translation[src[i+k]] = nn[k]

  # save translations
  trans_src = [src_sents[s] for s in translation.keys()]
  trans_trg = [trg_sents[t] for t in translation.values()]

  df = pd.DataFrame({'source sentences': trans_src, 'translations': trans_trg})
  #print(df)
  df.to_csv(os.path.join(args.output, 'cohere_translations.csv'), index=False)








def run_aligner():
    st.title("Cohere-Parallel-Language-Sentence-Alignment")

    # getting the API key
    cohere_api_key = 

    # Upload source and target files
    src_file = st.file_uploader("Upload source file", type=["txt"])
    trg_file = st.file_uploader("Upload target file", type=["txt"])

    # Run the aligner and display the output
    if st.button("Align"):
        if src_file is None or trg_file is None:
            st.warning("Please upload both source and target files.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save the uploaded files to the temporary directory
                src_file_path = os.path.join(tmpdir, "src.txt")
                with open(src_file_path, "wb") as f:
                    f.write(src_file.read())
                trg_file_path = os.path.join(tmpdir, "trg.txt")
                with open(trg_file_path, "wb") as f:
                    f.write(trg_file.read())

                # Set the path for the output file in the temporary directory
                output_file_path = os.path.join(tmpdir, "output.csv")

                # Run the aligner command
                command = [
                    "python3",
                    "-c",
                    "from streamlit_app import main; main()",
                    "--cohere_api_key", cohere_api_key,
                    "-m", "embed-multilingual-v2.0",
                    "-s", src_file_path,
                    "-t", trg_file_path,
                    "-o", output_file_path,
                    "--retrieval", "nn",
                    "--dot",
                    "--cuda"
                ]
                try:
                    result = subprocess.run(command, capture_output=True, cwd="/app/solid-doodle", text=True)
                except Exception as e:
                    st.error(f"Error running the aligner command: {e}")
                    return

                # Check if the command was successful and display the output
                if result.returncode == 0:
                    # Load the output file into a pandas dataframe
                    output_df = pd.read_csv(output_file_path)
                    st.dataframe(output_df)
                    
                    # Allow the user to download the output file as a text file
                    st.download_button(
                        "Download Output",
                        output_df.to_csv(index=False),
                        file_name="output.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Error running the aligner command. stdout: {result.stdout}, stderr: {result.stderr}")
# run the aligner function
run_aligner()
