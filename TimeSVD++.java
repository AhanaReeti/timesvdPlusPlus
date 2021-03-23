package fashiontrend;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import javafx.util.Pair;

public class svd {
	
	final int userNum = 6048000;  //number of users
	final int itemNum = 120000;   //number of items
	final int timeNum = 5115;    //number of days(time)
	final int binNum = 30;       //number of time bins
	final double AVG = 3.60073;  //average score
	double G_alpha = 0.00001;        //gamma for alpha
	final double L_alpha = 0.0004;   //learning rate for alpha
	final double L_pq = 0.015;       //learning rate for Pu & Qi
	double G = 0.007;                //general gamma
	final double Decay = 0.9;        //learning rate decay factor
	final double L = 0.005;          //general learning rate
	final int factor = 20;           //number of factors

	 double Bi[];
	 double Bu[];
	 double Qi[][];
	 double y[][];
	 double Pu[][] ;
	 double sumMW[][];
	 double Alpha_u[] ;
	 double Bi_Bin[][];
	 double Tu[];
	 
	 ArrayList<ArrayList<Pair <Pair<Integer, Integer>, Integer>>> train_data;
	 ArrayList<Integer> userIndex_data = new ArrayList<Integer>();
	 ArrayList<Integer> itemIndex_data = new ArrayList<Integer>();
	 ArrayList<Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>> test_data;
	 Vector<Map<Integer, Double>> Bu_t = new Vector<Map<Integer, Double>>();
	 Vector<Map<Integer, Double>> Dev = new Vector<Map<Integer,Double>>();
	 
	 
	 public double MyTrain() {
		 double preRmse = 1000;
		 int user, item, date;
		 double curRmse = 0.0;
		
		 for(int i=0;i<1000;i++) {
			 Train(); 
			 curRmse = Validate(AVG, Bu, Bi, Pu, Qi);
			 System.out.println("Rmse in step " + i + " : " + curRmse);
			 if(curRmse >= preRmse-0.00005){
		            break;
		        }
			 else{
		            preRmse = curRmse;
		        } 
		 } 
		 
		 return curRmse;
	 } 
	 
	 
	 
	 public void Train() {
		 
		 int userId, itemId, rating,time;
		 
		 for (userId = 0; userId < userNum; ++userId) {
			 int sz = train_data.get(userId).size();
			 double sqrtNum = 0;
			 double[] tmpSum = new double[factor];
		     for(int f=0; f<factor; f++){
		    	 tmpSum[f] = 0.0;
		     }
			
		     if (sz>1) sqrtNum = 1/(Math.sqrt(sz));
		     
		     for (int k = 0; k < factor; ++k) {
		            double sumy = 0;
		            for (int i = 0; i < sz; ++i) {
		                int itemI = train_data.get(userId).get(i).getKey().getKey();
		                sumy += y[itemI][k];
		            }
		            sumMW[userId][k] = sumy;
		        } 
		     
		    
			
		        for (int i = 0; i < sz; ++i) {
		            itemId = train_data.get(userId).get(i).getKey().getKey();
		            rating = train_data.get(userId).get(i).getKey().getValue();
		            time = train_data.get(userId).get(i).getValue();
		            double predict = predictScore(AVG, userId, itemId,time);
		            double error = rating - predict;
					
		            Bu[userId] += G * (error - L * Bu[userId]);
		            Bi[itemId] += G * (error - L * Bi[itemId]);
		            Bi_Bin[itemId][CalBin(time)] += G * (error - L * Bi_Bin[itemId][CalBin(time)]);
		            Alpha_u[userId] += G_alpha * (error * CalDev(userId,time)  - L_alpha * Alpha_u[userId]);
		            Bu_t.get(userId).put(time, G * (error - L * Bu_t.get(userId).get(time)) + Bu_t.get(userId).get(time));

		            for(int k=0;k<factor;k++){
		                double uf = Pu[userId][k];
		                double mf = Qi[itemId][k];
		                Pu[userId][k] += G * (error * mf - L_pq * uf);
		                Qi[itemId][k] += G * (error * (uf+sqrtNum*sumMW[userId][k]) - L_pq * mf);
		                tmpSum[k] += error*sqrtNum*mf;
		            } 
		        } 
		        
		        for (int j = 0; j < sz; ++j) {
		            itemId = train_data.get(userId).get(j).getKey().getKey();
		            for (int k = 0; k < factor; ++k) {
		                double tmpMW = y[itemId][k];
		                y[itemId][k] += G*(tmpSum[k]- L_pq *tmpMW);
		                sumMW[userId][k] += y[itemId][k] - tmpMW;
		            }
		        }
		   }
		 
		  for(userId = 0; userId < userNum; ++userId) {
		        int sz = train_data.get(userId).size();
		        double sqrtNum = 0;
		        if (sz>1) {
		        	sqrtNum = 1.0/Math.sqrt(sz);
		        }
		        for (int k = 0; k < factor; ++k) {
		            double sumy = 0;
		            for (int i = 0; i < sz; ++i) {
		                int itemI = train_data.get(userId).get(i).getKey().getKey();
		                sumy += y[itemI][k];
		            }
		            sumMW[userId][k] = sumy;
		        }
		    } 
		 
		  G *= Decay;
		  G_alpha *= Decay;
	  } 
	 
	 
	  public double Validate (double avg, double[] bu, double[] bi, double[][] pu, double[][] qi){
		 int userId, itemId;
		 int n = 0, rating, t;
		 double rmse = 0;
		 for (Pair testData : test_data){
			 userId = ((Pair<Integer, Integer>)testData.getKey()).getKey();
			 itemId = ((Pair<Integer, Integer>)testData.getKey()).getValue();
			 t  =  ((Pair<Integer, Integer>)testData.getValue()).getKey();
			 rating  = ((Pair<Integer, Integer>)testData.getValue()).getValue();
			 n++;
			 double pScore = predictScore(avg,userId,itemId,t);
			 rmse += (rating - pScore) * (rating - pScore);
		 }
		 
		 return Math.sqrt(rmse/n);   
	 }
		 

	    public double CalDev(int user, int timeArg) {
		   
		    if(Dev.elementAt(user).containsKey(timeArg)) {
		    	return Dev.elementAt(user).get(timeArg);
		    }
		    
		    double tmp = sign(timeArg - Tu[user]) * Math.pow((Math.abs(timeArg - Tu[user])), 0.4);
		    
		    Dev.elementAt(user).put(timeArg, tmp);
		    return tmp;
		}
	    
	    private int sign(double n) {
	    	return (n ==0)? 0 : ((n<0)?-1:1);
	    }
		 

		 public int CalBin(double timeArg){
			 int binsize = timeNum/binNum + 1;
			 return (int) (timeArg/binsize);
		 }
		 
		
		 public double predictScore(double avg, int userId, int itemId, int time){
			 
			 double tmp = 0.0;
			 int sz = train_data.get(userId).size();
			 double sqrtNum = 0;
			 if (sz>1){
				 sqrtNum = 1/(Math.sqrt(sz));
			 }
			 
			 for(int i=0;i<factor;i++){
				 tmp += (Pu[userId][i] +sumMW[userId][i]*sqrtNum) * Qi[itemId][i]; 
			 }
	   
			 double score = avg + Bu[userId] + Bi[itemId] + Bi_Bin[itemId][CalBin(time)] + Alpha_u[userId]*CalDev(userId,time) + Bu_t.elementAt(userId).get(time) + tmp;
			 if(score > 5){
			        score = 5;
			    }
			    if(score < 1){
			        score = 1;
			    }
			 return score;
		 } 
	
	 
	// SVD constructor and initialization
	public svd(double[] bi, double[] bu, int k, double[][] qi, double[][] pu) throws NumberFormatException, IOException {
		System.out.println("Contructing svd");
		resetEverything();
		train_data = new ArrayList<ArrayList<Pair<Pair<Integer, Integer>, Integer>>>();
		test_data = new ArrayList<Pair<Pair<Integer, Integer>, Pair<Integer,Integer>>>();
		
		for (int i=0; i<userNum; i++) {
	     	train_data.add(i, new ArrayList<Pair<Pair<Integer, Integer>, Integer>>());
	     }  
		
			 if(bi == null){
				 Bi = new double[itemNum];
			        for(int i=0;i<itemNum;i++){
			            Bi[i] = 0.0;
			        }
			    }
			 else{
				 Bi = bi;
			 }
			 
			 if(bu == null){
				 Bu = new double[userNum];
			        for(int i=0; i<userNum; i++) {
			            Bu[i] = 0.0;
			        }
			    }
			 else{
				 Bu = bu;
			 }
			 
			 Alpha_u = new double[userNum];
			 for(int i=0;i<userNum;i++){
			        Alpha_u[i] = 0.0;
			    }
			
			 Bi_Bin = new double [itemNum][];
			 for(int i=0;i<itemNum;i++){
			        Bi_Bin[i] = new double[binNum];
			    }
			 
			 for(int i=0;i<itemNum;i++){
			        for(int j=0;j<binNum;j++){
			            Bi_Bin[i][j] = 0.0;
			        }
			    }
			 
			 if(qi == null){
				 Qi = new double[itemNum][];
				 y = new double [itemNum][];
				 for(int i=0;i<itemNum;i++) {
			            Qi[i] = new double[factor];
			            y[i] = new double[factor];
			        }
				 for(int i=0;i<itemNum;i++) {
			         for(int j=0;j<factor;j++) {
			        	 // replace 32767 with java max rand number
			                Qi[i][j] = 0.1 * (Math.random() / (32767 + 1.0)) / Math.sqrt(factor);
			                y[i][j] = 0.0;
			            }
			        }
			 
			 }
			 else{
				 Qi = qi;
			    }
			 
			 if(pu == null){
				 sumMW = new double[userNum][];
				 Pu = new double[userNum][];
				 for(int i=0;i<userNum;i++){
			            Pu[i] = new double[factor];
			            sumMW[i] = new double[factor];
			        }
				 System.out.println("Contructing 17:");
				 for(int i=0;i<userNum;i++){
			            for(int j=0;j<factor;j++){
			            	// replace 32767 with java max rand number
			                sumMW[i][j] = 0.1 * (Math.random() / (32767 + 1.0)) / Math.sqrt(factor);
			                Pu[i][j] = 0.1 * (Math.random() / (32767 + 1.0)) / Math.sqrt(factor);
			            }
			           
			        }
			 }
			 else{
				 Pu = pu;
			   }
			 
			 
			 String csvFile = "C:\\Users\\udas3\\Documents\\thesisLibrary\\timesvd++\\amazonRating\\new\\ratings_Clothing_Shoes_and_Jewelry.csv";
		        String line = "";
		        String cvsSplitBy = ",";
		        int count = 0;

		        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
		        	Calendar cal = Calendar.getInstance();
		            while ((line = br.readLine()) != null) {
		                String[] ratingData = line.split(cvsSplitBy);
		                Date date = Date.from(Instant.ofEpochSecond(Integer.parseInt(ratingData[3])));
		                
		                cal.setTime(date);
		                int year = cal.get(Calendar.YEAR);
		                count++;
		                
		                train_data.get(Integer.parseInt(ratingData[0])).add(new Pair(new Pair(Integer.parseInt(ratingData[1]), Integer.parseInt(ratingData[2])), year));
		            }

		        } catch (IOException e) {
		            e.printStackTrace();
		        }
		        
		        String csvtestFile = "C:\\Users\\ahana\\Documents\\thesisLibrary\\timesvd++\\amazonRating\\ratings.csv";

		        try (BufferedReader br = new BufferedReader(new FileReader(csvtestFile))) {

		            while ((line = br.readLine()) != null) {
		                String[] ratingData = line.split(cvsSplitBy);
		                Date date = Date.from(Instant.ofEpochSecond(Integer.parseInt(ratingData[3])));
		                Calendar cal = Calendar.getInstance();
		                cal.setTime(date);
		                int year = cal.get(Calendar.YEAR);
		                
	                    test_data.add(new Pair(new Pair(Integer.parseInt(ratingData[0]), Integer.parseInt(ratingData[1])), new Pair(year, Integer.parseInt(ratingData[2]))));
		            }

		        } catch (IOException e) {
		            e.printStackTrace();
		        }
			 
			 
			 Tu = new double[userNum];
			 for(int i=0;i<userNum;i++){
			        double tmp = 0.0;
			        if(train_data.get(i).size()==0)
			        {
			            Tu[i] = 0.0;
			            continue;
			        }
			        for(int j=0;j<train_data.get(i).size();j++){
			            tmp += train_data.get(i).get(j).getValue();
			        }
			        Tu[i] = tmp/train_data.get(i).size();
			 }
			        
			 
			 for(int i=0;i<userNum;i++){
				 Map<Integer, Double> tmp = new HashMap<Integer, Double>();
			        for(int j=0;j<train_data.get(i).size();j++){			        		        		
			        	if (!tmp.containsKey(train_data.get(i).get(j).getValue())) {
			            	tmp.put(train_data.get(i).get(j).getValue(), 0.0000001);
			            }
			            else {
			            	continue;
			            }
			        }
			        Bu_t.add(tmp); 
			    }
			        
        
			     for(int l=0;l<userNum;l++){
			    	 Map<Integer, Double> tmp = new HashMap<Integer, Double>();
			            Dev.add(tmp);
			        }
			     System.out.println("Contructing svd finished");
			    }

	
	private void resetEverything() {
		Bi = null;
		Bu = null;
		Alpha_u = null;
		Tu = null;
	    Bi_Bin = null;
	    sumMW = null;
	    y = null;
	    Pu = null;
	    Qi = null;
	    train_data = null;
	    test_data = null;
	    userIndex_data = null;
	    itemIndex_data = null;
	}
				 
		
	}
	 

	 



