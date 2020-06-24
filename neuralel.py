import os
import sys
import copy
import pprint
import numpy as np
import tensorflow as tf
import json
import time 
import gc

from memory_profiler import profile
from readers.inference_reader import InferenceReader
from readers.test_reader import TestDataReader
from models.figer_model.el_model import ELModel
from readers.config import Config
from readers.vocabloader import VocabLoader
import readers.utils as utils 

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=7)

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("max_steps", 32000, "Maximum of iteration [450000]")
flags.DEFINE_integer("pretraining_steps", 32000, "Number of steps to run pretraining")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_string("model_path", "/fs/clip-quiz/naveen/neural_el/neural-el_resources/models/CD.model", "Path to trained model")
flags.DEFINE_string("dataset", "el-figer", "The name of dataset [ptb]")
flags.DEFINE_string("checkpoint_dir", "/tmp",
                    "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_integer("batch_size", 1, "Batch Size for training and testing")
flags.DEFINE_integer("word_embed_dim", 300, "Word Embedding Size")
flags.DEFINE_integer("context_encoded_dim", 100, "Context Encoded Dim")
flags.DEFINE_integer("context_encoder_num_layers", 1, "Num of Layers in context encoder network")
flags.DEFINE_integer("context_encoder_lstmsize", 100, "Size of context encoder hidden layer")
flags.DEFINE_integer("coherence_numlayers", 1, "Number of layers in the Coherence FF")
flags.DEFINE_integer("jointff_numlayers", 1, "Number of layers in the Coherence FF")
flags.DEFINE_integer("num_cand_entities", 30, "Num CrossWikis entity candidates")
flags.DEFINE_float("reg_constant", 0.00, "Regularization constant for NN weight regularization")
flags.DEFINE_float("dropout_keep_prob", 0.6, "Dropout Keep Probability")
flags.DEFINE_float("wordDropoutKeep", 0.6, "Word Dropout Keep Probability")
flags.DEFINE_float("cohDropoutKeep", 0.4, "Coherence Dropout Keep Probability")
flags.DEFINE_boolean("decoder_bool", True, "Decoder bool")
flags.DEFINE_string("mode", 'inference', "Mode to run")
flags.DEFINE_boolean("strict_context", False, "Strict Context exludes mention surface")
flags.DEFINE_boolean("pretrain_wordembed", True, "Use Word2Vec Embeddings")
flags.DEFINE_boolean("coherence", True, "Use Coherence")
flags.DEFINE_boolean("typing", True, "Perform joint typing")
flags.DEFINE_boolean("el", True, "Perform joint typing")
flags.DEFINE_boolean("textcontext", True, "Use text context from LSTM")
flags.DEFINE_boolean("useCNN", False, "Use wiki descp. CNN")
flags.DEFINE_boolean("glove", True, "Use Glove Embeddings")
flags.DEFINE_boolean("entyping", False, "Use Entity Type Prediction")
flags.DEFINE_integer("WDLength", 100, "Length of wiki description")
flags.DEFINE_integer("Fsize", 5, "For CNN filter size")

flags.DEFINE_string("optimizer", 'adam', "Optimizer to use. adagrad, adadelta or adam")

flags.DEFINE_string("config", 'configs/config.ini',
                    "VocabConfig Filepath")
flags.DEFINE_string("test_out_fp", "", "Write Test Prediction Data")

FLAGS = flags.FLAGS


from unidecode import unidecode

prog_start = time.time()

def FLAGS_check(FLAGS):
    if not (FLAGS.textcontext and FLAGS.coherence):
        print("*** Local and Document context required ***")
        sys.exit(0)
    assert os.path.exists(FLAGS.model_path), "Model path doesn't exist."

def decrypt(s):
    l = ""
    i = 0
    while i <len(s):
        if ord(s[i])< 128:
            l+=s[i]
            i+=1
        else:
            if len(unidecode(s[i]))>0:
                l+=unidecode(s[i])[0]
            else:
                l+="a"
            i+=1
    return l
def getCurrentMemoryUsage():
    ''' Memory usage in kB '''
    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    return int(memusage.strip())

@profile 
def main(_):
    pp.pprint(flags.FLAGS.__flags)

    output_file = "data/output.json"#sys.argv[2]

    range_start = 0#int(sys.argv[3])
    range_end = 10#int(sys.argv[4])

    file_name = "data/qanta.train.2018.04.18.json"#sys.argv[1]
    question_list = json.loads(open(file_name).read())["questions"]
    sentences = question_list[range_start:min(range_end,len(question_list))]

    FLAGS_check(FLAGS)

    config = Config(FLAGS.config, verbose=False)
    vocabloader = VocabLoader(config)

    print("Loading in variables!")
    word2idx, idx2word = vocabloader.getGloveWordVocab()
    wid2WikiTitle = vocabloader.getWID2Wikititle()
    crosswikis = utils.load(config.crosswikis_pruned_pkl)
    word2vec = vocabloader.loadGloveVectors()
    print("DONE LOADING IN VARIABLES!!!")
    
    all_entities = []

    for sent in sentences:
        tf.reset_default_graph()
        loc = config.test_file.replace("sampletest.txt","{}_{}.txt".format(range_start,range_end))
        w = open(loc,"w")
        config.test_file = loc
        sent["text"] = decrypt(sent["text"].replace("\xa0"," "))
        w.write(sent["text"].encode("ascii","ignore").decode("ascii"))
        print(sent["text"].encode("ascii","ignore").decode("ascii"))
        w.close()
        FLAGS.dropout_keep_prob = 1.0
        FLAGS.wordDropoutKeep = 1.0
        FLAGS.cohDropoutKeep = 1.0
        start = time.time()
        print("Test file {} ".format(config.test_file))
        reader = InferenceReader(config=config,
                                 vocabloader=vocabloader,
                                 test_mens_file=config.test_file,
                                 num_cands=FLAGS.num_cand_entities,
                                 batch_size=FLAGS.batch_size, word2idx=word2idx, idx2word=idx2word,
                                 wid2WikiTitle=wid2WikiTitle,crosswikis=crosswikis,word2vec=word2vec
                                 ,strict_context=FLAGS.strict_context,
                                 pretrain_wordembed=FLAGS.pretrain_wordembed,
                                 coherence=FLAGS.coherence)
        print("Took {} time to create inference reader".format(time.time()-start))
        docta = reader.ccgdoc
        model_mode = 'inference'

        config_proto = tf.ConfigProto()
        config_proto.allow_soft_placement = True
        config_proto.gpu_options.allow_growth=True
        sess = tf.Session(config=config_proto)
        
        print("COHSTR",reader.num_cohstr)

        """with sess.as_default():

            start = time.time()
            model = ELModel(
                sess=sess, reader=reader, dataset=FLAGS.dataset,
                max_steps=FLAGS.max_steps,
                pretrain_max_steps=FLAGS.pretraining_steps,
                word_embed_dim=FLAGS.word_embed_dim,
                context_encoded_dim=FLAGS.context_encoded_dim,
                context_encoder_num_layers=FLAGS.context_encoder_num_layers,
                context_encoder_lstmsize=FLAGS.context_encoder_lstmsize,
                coherence_numlayers=FLAGS.coherence_numlayers,
                jointff_numlayers=FLAGS.jointff_numlayers,
                learning_rate=FLAGS.learning_rate,
                dropout_keep_prob=FLAGS.dropout_keep_prob,
                reg_constant=FLAGS.reg_constant,
                checkpoint_dir=FLAGS.checkpoint_dir,
                optimizer=FLAGS.optimizer,
                mode=model_mode,
                strict=FLAGS.strict_context,
                pretrain_word_embed=FLAGS.pretrain_wordembed,
                typing=FLAGS.typing,
                el=FLAGS.el,
                coherence=FLAGS.coherence,
                textcontext=FLAGS.textcontext,
                useCNN=FLAGS.useCNN,
                WDLength=FLAGS.WDLength,
                Fsize=FLAGS.Fsize,
                entyping=FLAGS.entyping)

            print("Loading EL Model took {} time".format(time.time()-start))

            print("Doing inference")

            try:
                start = time.time()
                (predTypScNPmat_list,
                widIdxs_list,
                priorProbs_list,
                textProbs_list,
                jointProbs_list,
                evWTs_list,
                pred_TypeSetsList) = model.inference(ckptpath=FLAGS.model_path)
                print("Inference took {} time".format(time.time()-start))
            except:
                entity_list = {'qanta_id':sent['qanta_id'],'mentions':[]}
                all_entities.append(entity_list)
                print("No entities")
                continue
 
            start = time.time()
            numMentionsInference = len(widIdxs_list)
            numMentionsReader = 0
            for sent_idx in reader.sentidx2ners:
                numMentionsReader += len(reader.sentidx2ners[sent_idx])
            assert numMentionsInference == numMentionsReader

            mentionnum = 0
            entityTitleList = []
            print("Tokenized sentences {}".format(reader.sentences_tokenized))
            for sent_idx in reader.sentidx2ners:
                nerDicts = reader.sentidx2ners[sent_idx]
                sentence = ' '.join(reader.sentences_tokenized[sent_idx])
                for s, ner in nerDicts:
                    [evWTs, evWIDS, evProbs] = evWTs_list[mentionnum]
                    predTypes = pred_TypeSetsList[mentionnum]

                    entityTitleList.append(evWTs[2])
                    mentionnum += 1

            elview = copy.deepcopy(docta.view_dictionary['NER_CONLL'])
            elview.view_name = 'ENG_NEURAL_EL'
            for i, cons in enumerate(elview.cons_list):
                cons['label'] = entityTitleList[i]

            docta.view_dictionary['ENG_NEURAL_EL'] = elview

            print("Processing took {} time".format(time.time()-start))

            print("List of entities")
            #print(elview.cons_list)
            print("\n")
            
            s = sent["text"]
            print("New S is {}".format(s))
            e = elview.cons_list
            t = reader.sentences_tokenized 
            c = []
            f = []

            print(s)
            #print("E {}".format(e))
            print("T {}".format(t))

            for i in t:
                for j in i:
                    f.append(j)
            i = 0
            token_pointer = 0
            while token_pointer < len(f) and i < len(s):
                token_len = len(f[token_pointer])
                while i+token_len<len(s) and s[i:i+token_len] != f[token_pointer]:
                    i+=1
                c.append((i,token_len+i))
                i+=1
                token_pointer+=1
            if len(c) != len(f):
                print("ERROR in C and F")           
            unflattened_c = []
            c_pointer = 0
            for i in range(len(t)):
                l = c[c_pointer:c_pointer+len(t[i])]
                c_pointer+=len(t[i])
                unflattened_c.append(l)

            #print("C {}".format(c))
            #print("F {}".format(f))
            #print("Unflattened C {}".format(unflattened_c)) 

            entity_list = {'qanta_id':sent['qanta_id'],'mentions':[]}
            sentence_num = 0
               
            UNK = "<unk_wid>"
            for i in range(len(e)):
                if e[i]["label"]!=UNK:
                    all_words = False
                    while not all_words and sentence_num < len(t):
                        all_words = True
                        #print(e[i])
                        for word in range(e[i]["start"],e[i]["end"]+1):
                            if len(t[sentence_num])<=word or t[sentence_num][word] not in e[i]["tokens"]:
                                all_words = False
                        if not all_words:
                            sentence_num+=1
                    if sentence_num == len(t):
                        print("Error with sentence_num")
                    else:
                        entity_list['mentions'].append({'entity':e[i]["label"],'span':[unflattened_c[sentence_num][e[i]['start']][0],unflattened_c[sentence_num][e[i]['end']][1]]})
            #print("Entity list is {}".format(entity_list))

            all_entities.append(entity_list)
            local_vars = list(locals().items())
            del reader
     
            del predTypScNPmat_list
            del widIdxs_list
            del priorProbs_list
            del textProbs_list
            del jointProbs_list
            del evWTs_list
            del model
            del pred_TypeSetsList
            print("Memory usage {}".format(getCurrentMemoryUsage()))
            #print("All entities are {}".format(all_entities))
        del sess"""
        gc.collect()
        tf.reset_default_graph()

    w=open(output_file,"w")
    w.write(json.dumps(all_entities))
    w.close()

    print("Dumped JSON, all done")
    print("Took {} time".format(time.time()-prog_start))
    return 
    sys.exit()
if __name__ == '__main__':
    tf.app.run()
