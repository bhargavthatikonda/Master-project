package proj9052507_BPN;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import proj9052507_BPN.BackPropogation.NeuralNet;

public class RunTraining {

	public static void main(String[] args) throws Exception{
		Configuration conf = new Configuration();
		conf.set("mapred.job.queue.name","xxxx");
		conf.set("epoch","400");
		conf.set("inputColumns","17,18,19");
		conf.set("outputColumns","14");
		conf.set("nodesInHiddenLayers","4,3,1");
		Job job = new Job(conf);
		job.setJarByClass(RunTraining.class);
		job.setMapperClass(BackPropogationMapper.class);
		job.setCombinerClass(BackPropogationCombiner.class);
		job.setReducerClass(BackPropogationReducer.class);
		
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(TextArrayWriteable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		boolean success=job.waitForCompletion(true);
		if (success)
		{
			// Start predicting values
			String[] inputColumns = conf.get("inputColumns").split(",");
			String[] nodesInHiddenLayers = conf.get("nodesInHiddenLayers").split(",");
			int[] numNodesHidden=new int[nodesInHiddenLayers.length];
			int i = 0;
			for (String count:nodesInHiddenLayers)
			{
				numNodesHidden[i]=Integer.parseInt(count);
				i++;
			}
			BackPropogation neuralNet = new BackPropogation(inputColumns.length, numNodesHidden);
			neuralNet.getLayer(numNodesHidden.length-1).setIsSigmoid(false);
			
			Path recucerOutputPath = new Path(args[1]+"/part-r-00000");
			FileSystem fs = FileSystem.get(conf);			
			BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(recucerOutputPath)));
			String line;
			int layer=0;
            line=br.readLine();
            while (line != null){
            	String[] weights=line.split(",");
            	float[] wt=new float[weights.length-1];
            	for (int i2=1;i2<weights.length;i2++)
            	{
            		wt[i2-1]=Float.parseFloat(weights[i2]);
            	}
            	NeuralNet nt=neuralNet.getLayer(layer);
            	nt.setWeights(wt);
                line=br.readLine();
                layer++;
            }

            
            // Read input and predict values
			Path InputPath = new Path(args[2]);
			String[] inputColumnsPrediction = conf.get("inputColumns").split(",");
			BufferedReader br2=new BufferedReader(new InputStreamReader(fs.open(InputPath)));
			String line2;
			Path filenamePath = new Path(args[3]);
			try
			{
			    if (fs.exists(filenamePath))
			    {
			        fs.delete(filenamePath, true);
			    }
			}
			catch(Exception e)
			{
				throw e;
			}
			FSDataOutputStream fin = fs.create(filenamePath);
            line2=br2.readLine();
            float[] inputArray=null;
            while (line2 != null){
            	String[] inputs=line2.split(",");
            	inputArray=new float[inputColumnsPrediction.length];
    			for (int i1 = 0;i1<inputColumnsPrediction.length;i1++)
    			{
    				inputArray[i1]=Float.parseFloat(inputs[Integer.parseInt(inputColumnsPrediction[i1])-1]);
        			//fin.writeUTF("Value of Input number "+i1+" "+inputArray[i1]+"\n");
    			}
    			float[] prediction=neuralNet.run(inputArray);
    			for (int i1=0;i1<prediction.length;i1++)
    			{
    				fin.writeUTF(line2+","+prediction[i1]+"\n");
    				//fin.writeUTF("Value of Output number "+i1+" "+prediction[i1]+"\n");
    			}
    			//fin.writeUTF("***********************************************************\n");
    			
                line2=br2.readLine();
            }
            fin.close();

		}
		System.exit(success?0:1);
		
	}
}
