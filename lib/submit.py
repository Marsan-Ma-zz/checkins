import os, sys, time, pickle, gzip
from robobrowser import RoboBrowser

from lxml import html
from time import sleep

from datetime import datetime

#===========================================
#   Submitter
#===========================================  
class submitora(object):


  def __init__(self, username, password, competition='facebook-v-predicting-check-ins'):
    self.base = 'https://www.kaggle.com'
    self.login_url = '/'.join([self.base, 'account', 'login'])
    self.submit_url = '/'.join([self.base, 'c', competition, 'submissions', 'attach'])
    self.username = username
    self.password = password


  def submit(self, entry, message=None):
    browser = RoboBrowser(history=True)

    # [login]
    browser.open(self.login_url)
    login_form = browser.get_form(action='/account/login')
    login_form['UserName'].value = self.username
    login_form['Password'].value = self.password
    browser.submit_form(login_form)
    myname = browser.select('#header-account')[0].text
    print("[login] as \"%s\" @ %s" % (myname, datetime.now()))

    # [submit]
    browser.open(self.submit_url)
    submit_form = browser.get_form(action='/competitions/submissions/accept')
    submit_form['SubmissionUpload'].value = open(entry, 'r')
    if message: submit_form['SubmissionDescription'] = message
    browser.submit_form(submit_form)
    print("submitted @ %s" % datetime.now())
    score = browser.select(".my-submission")[0].select(".score")[0].text
    print("[result] score as %s @ %s" % (score, datetime.now()))
    return score


#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  sub = submitora(username='marsan@gmail.com', password='kaguya54')
  sub.submit(
    entry="/home/workspace/checkins/submit/blending_20160621_214954.csv",
    message="test2",
  )

