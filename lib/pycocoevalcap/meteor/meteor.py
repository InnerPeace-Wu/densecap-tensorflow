#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Modified by Linjie Yang for evaluating dense captioning
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res, imgIds=None):
        assert(gts.keys() == res.keys())
        if imgIds is None:
            imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)

            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        for i in range(0,len(imgIds)):
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        final_score = self.meteor_p.stdout.readline().strip()
        #print final_score
        score = float(final_score)
        self.lock.release()

        return score, scores


    def compute_score_m2m(self, gts, res, imgIds=None):
        assert(gts.keys() == res.keys())
        if imgIds is None:
            imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        tot_line = 0
        for i in imgIds:
            #assert(len(res[i]) == 1)
            for res_sent in res[i]:
                stat = self._stat(res_sent, gts[i])
                eval_line += ' ||| {}'.format(stat)
                tot_line += 1
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        for i in range(0,len(imgIds)):
            scores_im = []
            for j in xrange(len(res[i])):
                scores_im.append(float(self.meteor_p.stdout.readline().strip()))
            scores.append(scores_im)
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return score, scores
    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        return self.meteor_p.stdout.readline().strip()

    def score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats 
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score
 
    def __exit__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.wait()
        self.lock.release()
