package praktika2;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.sql.Date;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class kfCV {
	public static void main(String[] args) throws Exception{
		// 1. Argumentuen egiaztapena
        if (args.length < 2) {
            System.out.println("Erabilera: Java NaiveBayesEvaluation <data.arff> <emaitzak.txt>");
            return;
        }
        String arffPath = args[0];
        String outputPath = args[1];
        try {
        	// 2. Datuak kargatu
        	FileReader fitxategi = new FileReader(arffPath);
        	Instances data = new Instances(fitxategi);
        	fitxategi.close();
        	
        	// Zehaztu klasea
        	if (data.classIndex()==-1) data.setClassIndex(data.numAttributes()-1);
        	
        	// 3. Naive Bayes sailkatzailea eta Ebaluzioa (5-fCV)
        	NaiveBayes nB = new NaiveBayes();
        	Evaluation eval = new Evaluation(data);
        	
        	// 5-fold cross validation exekutatu
        	eval.crossValidateModel(nB, data, 5, new Random(1));
        	
        	//4. Emaitzak fitxategi batean idatzi
        	FileWriter fitxategiIdatzi = new FileWriter(outputPath);
        	PrintWriter idatzi = new PrintWriter(fitxategiIdatzi);
        	// Exekuzio data
        	DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
        	idatzi.write("Exekuzio data: " + dtf.format(LocalDateTime.now())); // LocalDateTime erabili
        	idatzi.println("Argumentuak: "+ arffPath+", "+outputPath);
        	idatzi.println("-------------------------");
        	// Nahasmen matrizea
        	idatzi.println("Naive Bayes 5-fold cross validation Nahasmen Matrizea: ");
        	idatzi.println(eval.toMatrixString());
        	
        	// Metrikak klase bakoitzeko eta batazbesteko ponderatua
        	idatzi.println("Metrikak: ");
        	for (int i=0;i<data.numClasses();i++) {
        		idatzi.printf("Klasea [%s]: %.4f%n", data.classAttribute().value(i), eval.precision(i));
        	}
        	idatzi.printf("Batazbesteko ponderatua: %.4f%n", eval.weightedPrecision());
        	
        	idatzi.close();
        }catch (Exception e) {
        	e.printStackTrace();	

        }
	}

}
