import os
from subprocess import call

#input_dirpath = '/data/fanfiction_ao3/annotated_10fandom/dev/fics_text_modified/'
input_dirpath = '/data/fanfiction_ao3/annotated_10fandom/dev/fics_text_tokenized/'
booknlp_log_dirpath = 'data/output/annotated_10fandom_dev/'
booknlp_output_dirpath = 'data/tokens/annotated_10fandom_dev/'

# cmd = './runjava novels/BookNLP -doc /data/fanfiction_ao3/annotated_10fandom/dev/fics_text/allmarvel_1621415 -p data/output/annotated_10fandom_dev/allmarvel_1621415 -tok data/tokens/allmarvel_1621415.tokens -f'

#booknlp_dirpath = '/usr0/home/mamille2/book-nlp/' # no whitespace tokenization
booknlp_dirpath = '/usr0/home/mamille2/book-nlp-whitespace-tok/' # no whitespace tokenization
os.chdir(booknlp_dirpath)

for fname in os.listdir(input_dirpath):
    cmd = ['./runjava', 'novels/BookNLP', '-doc', 
                os.path.join(input_dirpath, fname),
               '-p', os.path.join(booknlp_log_dirpath, os.path.splitext(fname)[0]),
               '-tok', f'{os.path.join(booknlp_output_dirpath, os.path.splitext(fname)[0])}.tokens', '-f']

    print(" ".join(cmd))
    call(cmd)
