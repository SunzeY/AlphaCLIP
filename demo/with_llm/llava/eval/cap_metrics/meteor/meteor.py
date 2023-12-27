# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import subprocess
import tarfile
import threading

import requests


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive."""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, 'wb') as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith('download_warning'):
            confirm_token = v

    if confirm_token:
        url = url + '&confirm=' + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


METEOR_GZ_URL = 'http://aimagelab.ing.unimore.it/speaksee/data/meteor.tgz'
METEOR_JAR = 'meteor-1.5.jar'


class Meteor:

    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        jar_path = os.path.join(base_path, METEOR_JAR)
        gz_path = os.path.join(base_path, os.path.basename(METEOR_GZ_URL))
        if not os.path.isfile(jar_path):
            if not os.path.isfile(gz_path):
                download_from_url(METEOR_GZ_URL, gz_path)
            tar = tarfile.open(gz_path, 'r')
            tar.extractall(path=os.path.dirname(os.path.abspath(__file__)))
            tar.close()
            os.remove(gz_path)

        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert (len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
        self.meteor_p.stdin.flush()
        for i in range(0, len(imgIds)):
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return score, scores

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(
            ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
        self.meteor_p.stdin.flush()
        raw = self.meteor_p.stdout.readline().decode().strip()
        numbers = [str(int(float(n))) for n in raw.split()]
        return ' '.join(numbers)

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()

    def __str__(self):
        return 'METEOR'
