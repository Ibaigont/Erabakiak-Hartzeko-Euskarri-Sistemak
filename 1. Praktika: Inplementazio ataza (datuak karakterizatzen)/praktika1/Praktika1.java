package praktika1;

import java.io.FileReader;
import weka.core.*;

public class Praktika1 {
	public static void main(String[] args) throws Exception{
		// 1. Lehenik eta behin datuak kargatu
		String path = "/home/ubuntu/Downloads/heart-c.arff";
		System.out.println("Fitxategiaren path-a: "+ path);
		FileReader fitxategi = new FileReader(path);
		Instances data = new Instances(fitxategi);
		fitxategi.close();
		
		// 2. Klase atributua ezarri
		data.setClassIndex(data.numAttributes()-1);
		
		// 3. Instantzia eta atributu kopurua
		int instantziaKop = data.numInstances();
		int atributuKop = data.numAttributes();
		System.out.println("Instantzia kopurua: "+ instantziaKop);
		System.out.println("Atributu kopurua: "+ atributuKop);
		
		// 4. Lehen atributuaren balio desberidon kopurua edo distinct values
		AttributeStats statsLehena = data.attributeStats(0);
		int statKop = statsLehena.distinctCount;
		System.out.println("Lehenengo atributuaren balio desberdin kopurua: " + statKop);
		
		// 5. Azken atributuak hartzen dituen balioak, maiztasunak eta klase minoritarioa
		int klaseIndex = data.classIndex();
		AttributeStats statsKlasea = data.attributeStats(klaseIndex);
		System.out.println("Klasearen maiztazunak:");
		int minMaiztasuna = Integer.MAX_VALUE;
		String klaseMinoritatioa= "";
		for (int i=0;i<data.attribute(klaseIndex).numValues();i++) {
			String balioIzena = data.attribute(klaseIndex).value(i);
			int maiztasuna = statsKlasea.nominalCounts[i];
			System.out.println("-"+balioIzena+":"+maiztasuna);
			if (maiztasuna< minMaiztasuna) {
				minMaiztasuna = maiztasuna;
				klaseMinoritatioa = balioIzena;
			}
		}
		System.out.println("Klase minoritarioa: "+ klaseMinoritatioa);
		
		// 6. Azken aurreko atributuaren missing value kopurua
		int azkenAurrekoIndex = data.numAttributes()-2;
		AttributeStats azkenAurrekoStats = data.attributeStats(azkenAurrekoIndex);
		int missingKop = azkenAurrekoStats.missingCount;
		System.out.println("Azken aurreko atrinbutuaren missing kopurua: "+ missingKop);
	}
}
