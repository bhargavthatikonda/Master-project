package proj9052507_BPN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;

public class BackPropogationCombiner extends Reducer<Text,TextArrayWriteable,Text,TextArrayWriteable>{

	private Text word = new Text();
	private TextArrayWriteable weight;

	public void reduce(Text key,Iterable<TextArrayWriteable> vals,Context context) throws IOException,InterruptedException
	{
		Configuration conf = context.getConfiguration();
		int epoch = Integer.parseInt(conf.get("epoch"));
		String[] inputColumns = conf.get("inputColumns").split(",");
		String[] outputColumns = conf.get("outputColumns").split(",");
		String[] nodesInHiddenLayers = conf.get("nodesInHiddenLayers").split(",");
		int[] numNodesHidden=new int[nodesInHiddenLayers.length];
		float [][] weights=new float[nodesInHiddenLayers.length][];

		int i = 0;
		for (String count:nodesInHiddenLayers)
		{
			numNodesHidden[i]=Integer.parseInt(count);
			weights[i]=new float[numNodesHidden[i]];
			i++;
		}

		Iterator<TextArrayWriteable> it = vals.iterator();
		ArrayList<ArrayList<String>> recordArrayList=new ArrayList<ArrayList<String>>();
		int numOfRecords=0;
		while (it.hasNext()) {
			numOfRecords++;
			ArrayList<String> columns=new ArrayList<String>(); 
			Writable[] tmp=it.next().get();
			for(Writable txt:tmp)
			{
				columns.add(txt.toString());
			}
			recordArrayList.add(columns);
		}

		float[][] trainingInputs = new float[numOfRecords][inputColumns.length];
		float[][] trainingOutputs = new float[numOfRecords][outputColumns.length];
		float[][] trainingErrors = new float[numOfRecords][outputColumns.length];

		i=0;
		for(int i1=0;i1<numOfRecords;i1++)
		{
			ArrayList<String> recordString = recordArrayList.get(i1);
			
			int j=0;
			for(String columnNumber:inputColumns)
			{
				trainingInputs[i1][j] = Float.parseFloat(recordString.get(Integer.parseInt(columnNumber)-1));
				j++;
			}

			j=0;
			for(String columnNumber:outputColumns)
			{
				trainingOutputs[i1][j] =Float.parseFloat(recordString.get(Integer.parseInt(columnNumber)-1));
				j++;				
			}

		}


		BackPropogation neuralNet = new BackPropogation(inputColumns.length, numNodesHidden);
		neuralNet.getLayer(numNodesHidden.length-1).setIsSigmoid(false);
		Random r = new Random();

		for(int loopCount = 0; loopCount <= epoch; loopCount++)
		{
			for (int i1 = 0; i1 < trainingOutputs.length; i1++) {
				int index = r.nextInt(trainingOutputs.length);
				neuralNet.train(trainingInputs[index], trainingOutputs[index], 0.5f, 0.4f);
			}

			for (int i1 = 0; i1 < trainingOutputs.length; i1++) {
				float[] t = new float[trainingInputs[i1].length]; 
				t=trainingInputs[i1];
				float[] predictedOutput=new float[trainingOutputs.length];
				predictedOutput=neuralNet.run(t);
				for(int numOutNode=0; numOutNode<predictedOutput.length;numOutNode++)
				{
					trainingErrors[i1][numOutNode]=0.5f * (float) Math.pow(trainingOutputs[i1][numOutNode]-predictedOutput[numOutNode],2); 
				}
			}

			float[] totalError = new float[outputColumns.length];
			for (int i1=0; i1 < outputColumns.length;i1++){
				for (int j=0;j<trainingOutputs.length;j++){
					totalError[i1]+=trainingErrors[j][i1];
				}
			}

		}

		weights=neuralNet.getWeights();
		String[] strweight=new String[neuralNet.layers.length];
		for (int i1=0;i1<neuralNet.layers.length;i1++)
		{
			strweight[i1]="";
			for (int i2=0;i2<weights[i1].length;i2++)
			{
				strweight[i1]+=weights[i1][i2]+",";
			}
			
			// Number of input records passed as last value in case weighted average is required
			strweight[i1]+=Integer.toString(numOfRecords);
			word.set("layer"+i1);
			String[] stringArray={strweight[i1]};
			weight = new TextArrayWriteable(stringArray);
			context.write(word,weight);
		}


	}
}
