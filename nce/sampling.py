import math
import torch
import random

def binary_search(vector, left, right, target):
  # 0.3  0.6  0.8      0.9   1.0
  # 0    1    2        3     4
  #      target  0,85
  #      return idx 4
  # return idx such that vector[idx - 1] <= target < vector[idx]
  if target < vector[left][-1]:
    return 0

  while left < right - 1:
    mid = int((left + right) / 2)
    if vector[mid][-1] > target:
      right = mid
    elif vector[mid][-1] == target:
      idx =  mid 
      return idx
    else:
      left = mid
  idx = left + 1
#  assert(vector[idx - 1][1] < target and target <= vector[idx][1])
  return idx

def binary_search_2(vector, left, right, target):
  # 0.3  0.6  0.8      0.9   1.0
  # 0    1    2        3     4
  #      target  0,85
  #      return idx 4
  # return idx such that vector[idx - 1] <= target < vector[idx]
  if target < vector[left]:
    return 0

  while left < right - 1:
    mid = int((left + right) / 2)
    if vector[mid] > target:
      right = mid
    elif vector[mid] == target:
      idx =  mid 
      return idx
    else:
      left = mid
  idx = left + 1
#  assert(vector[idx - 1][1] < target and target <= vector[idx][1])
  return idx


class AliasMultinomial(torch.nn.Module):
    '''Alias sampling method to speedup multinomial sampling

    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.

    Attributes:
        - probs: the probability density of desired multinomial distribution

    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        self.probs = probs / probs.sum()

        cpu_probs = probs.cpu()
        K = len(probs)

        # such a name helps to avoid the namespace check for nn.Module
        self_prob = [0] * K
        self_alias = [0] * K

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial

        Args:
            - size: the output size of samples
        """

        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        ret = (oq + oj).view(size)
        return ret


class BandedSampler(torch.nn.Module):
  def __init__(self, probs, num_samples):
    super(BandedSampler, self).__init__()
    self.probs = probs / probs.sum()
    cpu_probs = probs.cpu()
    K = len(probs)

    self.sampling_probs = probs * num_samples
    self.num_samples = num_samples
    self.inclusion_probs = self.probs * self.num_samples

    self.normalize()
    self.shuffle()
    self.counter = 0

    self.inclusion_probs = self.inclusion_probs.cuda()

  def normalize(self):
    if self.num_samples == len(self.probs):
      self.inclusion_probs = self.inclusion_probs * 0.0 + 1.0
      self.inclusion_probs_cpu = self.inclusion_probs.cpu()
      return

    while self.inclusion_probs.max() > 1.0:
      sum_before = self.inclusion_probs.sum()
      self.inclusion_probs.clamp_(0.0, 1.0)
      sum_after = self.inclusion_probs.sum()
      self.inclusion_probs = torch.exp(torch.log(self.inclusion_probs) + math.log(sum_before / sum_after))
      self.inclusion_probs *= self.num_samples / self.inclusion_probs.sum()
    self.inclusion_probs_cpu = self.inclusion_probs.cpu()
    self.inclusion_probs_cpu *= self.num_samples / self.inclusion_probs_cpu.sum()
#    print (self.inclusion_probs.sum())
#    print (self.inclusion_probs_cpu.sum())

  def shuffle(self):
    cpu_t_list = self.inclusion_probs_cpu.numpy().tolist()
    pairs = []
    for i, b in enumerate(cpu_t_list):
      pairs.append([i, b])
    random.shuffle(pairs)

    self.bands = [] # t is 3d and t[i][j] is a pair [idx, cumulative_prob_so_far]
    cur = [] # cur is 2d and cur[i] is i'th [idx, cum] pair
    cur_sum = 0.0
    for idx, prob in pairs:
      if cur_sum + prob <= 1.0:
        cur_sum = cur_sum + prob
        cur.append([idx, cur_sum])
      elif cur_sum < 1.0:
        cur.append([idx, 1.0])
        self.bands.append(cur)
        cur_sum = cur_sum + prob - 1.0
        cur = [[idx, cur_sum]]
      else:
        self.bands.append(cur)
        cur_sum = cur_sum + prob - 1.0
        cur = [[idx, cur_sum]]

    self.bands.append(cur)

    # solving floating point problems
    if len(self.bands) > self.num_samples:
      # there is one more column than expected, add the extra to the previous
      for i in range(len(self.bands[-1])):
        self.bands[-1][i][1] += 1.0
      self.bands[-2] += self.bands[-1]
      del self.bands[-1]

    # renormalize last column
    s = self.bands[-1][-1][1]
    for i in range(len(self.bands[-1])):
      self.bands[-1][i][1] /= s

  def draw(self, *size):
    self.counter += 1
    if self.counter % 1000 == 0:
      self.shuffle()
    rand = random.uniform(0, 1)
    samples = []
    for pairs in self.bands:
      # i is 2d, a list of [idx, cumu] pairs
      idx = binary_search(pairs, 0, len(pairs), rand)
      samples.append(pairs[idx][0])
    samples.sort()
    return torch.LongTensor(samples).view(size)

class BandedGroupSampler(torch.nn.Module):
  def __init__(self, probs, num_samples):
    super(BandedGroupSampler, self).__init__()
    self.probs = probs / probs.sum()
    cpu_probs = probs.cpu()
    K = len(probs)

    self.sampling_probs = probs * num_samples
    self.num_samples = num_samples
    self.inclusion_probs = self.probs * self.num_samples

    self.normalize()
    self.grouping()
    self.shuffle()
    self.counter = 0

  def normalize(self):
    if self.num_samples == len(self.probs):
      self.inclusion_probs = self.inclusion_probs * 0.0 + 1.0
      self.inclusion_probs_cpu = self.inclusion_probs.cpu()
      return

    while self.inclusion_probs.max() > 1.0:
      sum_before = self.inclusion_probs.sum()
      self.inclusion_probs.clamp_(0.0, 1.0)
      sum_after = self.inclusion_probs.sum()
      self.inclusion_probs = torch.exp(torch.log(self.inclusion_probs) + math.log(sum_before / sum_after))
      self.inclusion_probs *= self.num_samples / self.inclusion_probs.sum()
    self.inclusion_probs_cpu = self.inclusion_probs.cpu()
#    self.inclusion_probs_cpu *= self.num_samples / self.inclusion_probs_cpu.sum()
    assert(self.inclusion_probs_cpu.max() <= 1.0)

  def grouping(self):
    self.groups = []
    self.CDFs = []
    # group[i] is a list [a, b, sum], where [a, b) an index interval to self.inclusion_probs_cpu
    # and sum is the total prob

    cur_sum = 0.0
    left = 0
#    tmp = 0.0
    for right in range(len(self.inclusion_probs_cpu)):
      this_prob = self.inclusion_probs_cpu[right].item()
      new_sum = cur_sum + this_prob
      if new_sum > 1.0:
        self.groups.append([left, right, cur_sum])
#        tmp += cur_sum
        cur_sum = this_prob
        left = right
      else:
        cur_sum = new_sum
    self.groups.append([left, len(self.inclusion_probs_cpu), cur_sum])

    for i in range(len(self.groups)):
      cdf = []
      cur = 0.0
      for j in range(self.groups[i][0], self.groups[i][1]):
        p = self.inclusion_probs_cpu[j].item() # / self.groups[i][2]
        cdf.append(p + cur)
        cur = p + cur
#      print (cdf)
      self.CDFs.append(cdf)

  def shuffle(self):
    pairs = []
    for i, b in enumerate(self.groups):
      pairs.append([i, b[2]])
    random.shuffle(pairs)

    self.bands = [] # t is 3d and t[i][j] is a pair [idx, cumulative_prob_so_far]
    cur = [] # cur is 2d and cur[i] is i'th [idx, cum] pair
    cur_sum = 0.0
    for idx, prob in pairs:
      if cur_sum + prob <= 1.0:
        cur_sum = cur_sum + prob
        cur.append([idx, cur_sum])
      elif cur_sum < 1.0:
        cur.append([idx, 1.0])
        self.bands.append(cur)
        cur_sum = cur_sum + prob - 1.0
        cur = [[idx, cur_sum]]
      else:
        self.bands.append(cur)
        cur_sum = cur_sum + prob - 1.0
        cur = [[idx, cur_sum]]

    self.bands.append(cur)

    # solving floating point problems
    if len(self.bands) > self.num_samples:
      # there is one more column than expected, add the extra to the previous
      for i in range(len(self.bands[-1])):
        self.bands[-1][i][1] += 1.0
      self.bands[-2] += self.bands[-1]
      del self.bands[-1]

    # renormalize last column
    s = self.bands[-1][-1][1]
    for i in range(len(self.bands[-1])):
      self.bands[-1][i][1] /= s

  def draw(self, *size):
    self.counter += 1
    if self.counter % 1000 == 0:
      self.shuffle()
    rand = random.uniform(0, 1)
    samples = []
    for pairs in self.bands:
      # i is 2d, a list of [idx, cumu] pairs
      pair_idx = binary_search(pairs, 0, len(pairs), rand)
#      print()
#      print (pairs, pair_idx)
      group_idx = pairs[pair_idx][0]
      idx = self.get_random_one(group_idx)
      samples.append(self.groups[group_idx][0] + idx)
    samples.sort()
    return torch.LongTensor(samples).view(size)

  def get_random_one(self, group_idx):
    rand = random.uniform(0, self.groups[group_idx][2])
    idx = binary_search_2(self.CDFs[group_idx], 0, len(self.CDFs[group_idx]), rand)
    return idx
