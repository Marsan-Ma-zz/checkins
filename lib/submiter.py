import os, re, sys, time, pickle, gzip
from robobrowser import RoboBrowser

from lxml import html
from time import sleep

from datetime import datetime

#===========================================
#   Submitter
#===========================================  
class submiter(object):


  def __init__(self, competition='facebook-v-predicting-check-ins'):
    self.base = 'https://www.kaggle.com'
    self.login_url = '/'.join([self.base, 'account', 'login'])
    self.submit_url = '/'.join([self.base, 'c', competition, 'submissions', 'attach'])
    self.submit_dashboard = '/'.join([self.base, 'c', competition, 'submissions'])

  def read_account(self):
    if os.path.exists("./.kaggle"):
      self.username, self.password = open("./.kaggle", 'rt').readline().replace("\n",'').split(',')
    else:
      print("./.kaggle not exists, cannot auto-submit!")
      self.username, self.password = None, None


  def submit(self, entry, message=None):
    browser = RoboBrowser(history=True)
    self.read_account()

    # [login]
    browser.open(self.login_url)
    login_form = browser.get_form(action='/account/login')
    login_form['UserName'].value = self.username
    login_form['Password'].value = self.password
    browser.submit_form(login_form)
    myname = browser.select('#header-account')[0].text
    print("[SUBMIT] login as \"%s\" @ %s" % (myname, datetime.now()))

    # [submit]
    browser.open(self.submit_url)
    submit_form = browser.get_form(action='/competitions/submissions/accept')
    submit_form['SubmissionUpload'].value = open(entry, 'r')
    if message: submit_form['SubmissionDescription'] = str(message)
    browser.submit_form(submit_form)
    print("[SUBMIT] submitted @ %s" % datetime.now())

    # [receive score]
    # sleep(300)
    # fname = entry.split('/')[-1]
    # print("finding %s" % fname)
    # score = None
    # fmt = re.compile('\d+(\.\d+)?')
    # while not score:
    #   browser.open(self.submit_dashboard)
    #   for tr in browser.select(".submissions tr"):
    #     tds = tr.select("td")
    #     if (len(tds) >= 4) and (fname in tds[1].text):
    #       score = tds[2].text.strip()
    #       break
    #       # try:
    #       #   score = float(tds[2].text.strip())
    #         # break
    #       # except ValueError:
    #       #   pass
    #   if not score:
    #     print("%s not found, waiting 10 secs ..." % fname)
    #     sleep(10)
    # print("[SUBMIT] result score as %s @ %s" % (score, datetime.now()))
    # return score


#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  sub = submiter()
  sub.submit(
    entry="/home/workspace/checkins/submit/skrf_cfeats_1500tree_m_split_7.csv",
    message="skrf 1500trees min split=7 try.",
  )

