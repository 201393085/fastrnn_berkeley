package edu.berkeley.nlp.lm.webserver;

import edu.berkeley.nlp.lm.FasterRnnlm.NNet;
import edu.berkeley.nlp.lm.FasterRnnlm.Vocabulary;
import edu.berkeley.nlp.lm.NgramLanguageModel;
import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.handler.AbstractHandler;
import org.json.JSONArray;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * Created by nlp on 16-12-2.
 */
public class QueryHandler extends AbstractHandler {
    private static String strClassName = QueryHandler.class.getName();
    private static Logger logger = Logger.getLogger(strClassName);
    private NgramLanguageModel lm;
    private NNet nnet;
    private Vocabulary vocab;
    private double lambda;

    public QueryHandler(NgramLanguageModel lm, NNet nnet, Vocabulary vocab, double lambda){
        this.lm = lm;
        this.nnet = nnet;
        this.vocab = vocab;
        this.lambda = lambda;
    }
    @Override
    public void handle(String target, Request baseRequest, HttpServletRequest request,

                       HttpServletResponse response) throws IOException, ServletException {

        response.setContentType("text/plain;charset=utf-8");

        response.setStatus(HttpServletResponse.SC_OK);

        baseRequest.setHandled(true);

        final String text = request.getParameter("postSenText");

        try{
            if(text == null){
                response.getWriter().println("{}");
            }else{
                boolean kNCEAccurate = false;
                int cnt = 0;

                JSONArray jsonArray = new JSONArray(text);
                JSONArray ans = new JSONArray();
                for(int i = 0; i < jsonArray.length(); ++i) {
                    String s = jsonArray.getString(i);
                    int [] sen = vocab.GetSentenceIndices(s);
                    List<String> words = Arrays.asList(s.trim().split("\\s+"));
                    double s1, s2;
                    if(sen ==null) s1 = 0;
                    else s1 = nnet.EvaluateLM(sen, kNCEAccurate)/log(2);
                    cnt+=words.size();
                    s2 = lm.scoreSentence(words);
                    ans.put((lambda)*s1+(1-lambda)*s2);
                //    logger.info(s1+" "+s2+" "+((lambda)*s1+(1-lambda)*s2));
                    logger.info(s+" "+new Double((lambda)*s1+(1-lambda)*s2).toString());
                }
                response.getWriter().print(ans);
            }
        }catch(Exception e) {
       //     logger.error("", e);
            response.getWriter().println("{}");
        }

        return;
    }
}
