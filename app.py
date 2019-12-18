from transformers import GPT2LMHeadModel, GPT2Tokenizer
from argparse import ArgumentParser
import torch
from pprint import pformat
import random
from flask import Flask, render_template, request, jsonify, redirect, url_for 
import os 
import logging
import yaml 
# import mysql.connector 

from utils import sample_sequence, add_special_tokens_ 

app = Flask(__name__)
app.secret_key = 'justin12'

model = None 
tokenizer = None 
args = None 
db = None 

def save_to_db(title: str, output: str) -> None:
  
  cursor = db.cursor() 
  sql = "INSERT INTO main (title, article) VALUES (%s, %s)"
  val = (title, output)
  cursor.execute(sql, val) 
  db.commit() 
  
  cursor.close() 

@app.route('/ased_api', methods=['GET'])
def api(): 
  """ Handle request and output model score in json format"""
  if args ==None: 
    initialize()

  history_text = None 

  # Handle GET requests: 
  if request.method == "GET": 
    if request.args: 
      history_text = request.args.getlist("input")
      print(history_text)

  if history_text is not None: 
    print(f"Received valid request through API - \"input\": {history_text}")
  else: 
    return jsonify({"error": "Invalid JSON request. Provide GET request as {\"input\": \"<your dialogue history as list>\"}"})

  personality = [] 
  history = [tokenizer.encode(t) for t in history_text]
  with torch.no_grad():
      out_ids = sample_sequence(personality, history, tokenizer, model, args, current_output=None)
  out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

  # save_to_db(title, out_text)

  return jsonify({"history": history_text, "response": out_text})

def initialize(): 
  global args
  global model 
  global tokenizer 
  global db

  # initialize args 
  config = yaml.safe_load(open('config/config.yaml', 'r'))
  args = config['default']
  args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__file__)
  logger.info(pformat(args))

  # initialize model and tokenizer 
  logger.info("Get pretrained model and tokenizer")
  model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
  tokenizer = tokenizer_class.from_pretrained(args['model_checkpoint'])
  model = model_class.from_pretrained(args['model_checkpoint'])
  model.to(args['device'])
  model.eval()
  add_special_tokens_(model, tokenizer)

  # connect to database 
  # db_config = config['mysql']
  # db = mysql.connector.connect(
  #   host=db_config['host'], 
  #   user=db_config['user'],
  #   passwd=db_config['passwd'], 
  #   database=db_config['database']
  # )

  logger.info("Initialization of model and tokenizer complete.")

if args is None: 
  initialize()


if __name__ == '__main__':
  # app.run(host='0.0.0.0', port=5000, use_debugger=False, use_reloader=False, passthrough_errors=True)
  app.run(host='0.0.0.0', port=5000, debug=True)
