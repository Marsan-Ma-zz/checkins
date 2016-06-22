import os, sys, time, pickle, gzip
from robobrowser import RoboBrowser

from lxml import html
from time import sleep

from datetime import datetime

#===========================================
#   Submitter
#===========================================  
class submitor(object):


  def __init__(self, competition='facebook-v-predicting-check-ins'):
    self.base = 'https://www.kaggle.com'
    self.login_url = '/'.join([self.base, 'account', 'login'])
    self.submit_url = '/'.join([self.base, 'c', competition, 'submissions', 'attach'])
    
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
    # score = browser.select(".my-submission")[0].select(".score")[0].text
    for i in range(10):
      score = browser.select('.submission-results strong')
      if score:
        print(score)
    print("[SUBMIT] result score as %s @ %s" % (score, datetime.now()))
    return score


#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  sub = submitor()
  sub.submit(
    entry="/home/workspace/checkins/submit/submit_skrf_submit_full_20160621_234034_0.0000.csv",
    message="",
  )

