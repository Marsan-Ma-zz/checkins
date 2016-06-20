import os, sys, time, pickle, gzip
from mechanize import Browser
from lxml import html
from time import sleep

from datetime import datetime

#===========================================
#   Submitter
#===========================================  
def submitor(object):

  def __init__(username, password, competition='facebook-v-predicting-check-ins'):
    self.base = 'https://www.kaggle.com'
    self.submit_url = '/'.join([self.base, 'c', competition, 'submissions', 'attach'])
    self.username = username
    self.password = password


  def submit(entry, message=None):
    browser = Browser()
    browser.open(self.base)
    browser.select_form(nr=0)
    browser['UserName'] = self.username
    browser['Password'] = self.password
    browser.submit()
    browser.open(self.submit_url)
    browser.select_form(nr=0)
    browser.form.add_file(open(entry), filename=entry)

    if message:
        browser['SubmissionDescription'] = message

    browser.submit()

    while True:
      leaderboard = html.fromstring(browser.response().read())
      score = leaderboard.cssselect('.submission-results strong')

      if len(score) and score[0].text_content():
          score = score[0].text_content()
          break

      sleep(30)
      browser.reload()

