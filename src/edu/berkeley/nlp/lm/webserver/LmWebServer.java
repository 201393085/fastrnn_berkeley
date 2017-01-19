package edu.berkeley.nlp.lm.webserver;

import edu.berkeley.nlp.lm.FasterRnnlm.NNet;
import edu.berkeley.nlp.lm.FasterRnnlm.Vocabulary;
import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.io.LmReaders;
import org.eclipse.jetty.server.Connector;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.util.thread.QueuedThreadPool;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Created by nlp on 16-12-2.
 */
public class LmWebServer {
    private static String strClassName = LmWebServer.class.getName();
    private static Logger logger = Logger.getLogger(strClassName);
    private static Server server;

    private static int DEFAULT_PORT = 55555;
    private static final int DEFAULT_MAX_IDLE_TIME = 3000;
    private static final int DEFAULT_MAX_THREAD_NUM = 20;
    private static final int DEFAULT_MIN_THREAD_NUM = 4;

    private static NgramLanguageModel lm;
    private static NNet nnet;
    private static Vocabulary vocab;
    private static double lambda;

    public static Server initializeServer(){
        QueuedThreadPool threadPool = null;
        threadPool = new QueuedThreadPool(DEFAULT_MAX_THREAD_NUM, DEFAULT_MIN_THREAD_NUM);
        Server server = new Server(threadPool);

        //Connector
        ServerConnector connector = new ServerConnector(server);
        connector.setPort(DEFAULT_PORT);
        connector.setIdleTimeout(DEFAULT_MAX_IDLE_TIME);
        server.setConnectors(new Connector[]{connector});
        //Handler
        server.setHandler(new QueryHandler(lm,nnet,vocab,lambda));

        return server;
    }

    private static void usage() {
        System.err.println("Usage: <Berkeley LM binary file> <outputfile>*\nor\n-g <vocab_cs file> <Google LM Binary>");
        System.exit(1);
    }
    private static NgramLanguageModel<String> readBinary(boolean isGoogleBinary, String vocabFile, String binaryFile) {
        NgramLanguageModel<String> lm = null;
        if (isGoogleBinary) {
            edu.berkeley.nlp.lm.util.Logger.startTrack("Reading Google Binary " + binaryFile + " with vocab " + vocabFile);
            lm = LmReaders.readGoogleLmBinary(binaryFile, vocabFile);
            edu.berkeley.nlp.lm.util.Logger.endTrack();
        } else {
            edu.berkeley.nlp.lm.util.Logger.startTrack("Reading LM Binary " + binaryFile);
            lm = LmReaders.readLmBinary(binaryFile);
            edu.berkeley.nlp.lm.util.Logger.endTrack();
        }
        return lm;
    }
    public static void main(String[] argv) throws IOException {
        String vocab_path = argv[0];
        String fpath = argv[1];
        String bpath = argv[2];
        String Lambda = argv[3];
        String Port = argv[4];

        DEFAULT_PORT = Integer.valueOf(Port);
        lambda = Double.valueOf(Lambda);

        logger.info("reading berkeley:");
        lm = readBinary(false, null, bpath);
        logger.info("loading fastrnn vocabulary:");
        vocab = Vocabulary.Load(vocab_path);
        nnet = new NNet(vocab,fpath);

        server = initializeServer();
        if(server == null){
            logger.severe("Sentence Segmentation Handler Initialization Failed!");
            return;
        }

        try {
            server.start();
            server.join();
        } catch (Exception e) {
            logger.severe(e.toString());
            return;
        }
    }
}
