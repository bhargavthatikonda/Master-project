package proj9052507_BPN;

import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;

public class BackPropogationReducer extends Reducer<Text,TextArrayWriteable,Text,Text>{

	private Text word = new Text();
	private Text value = new Text();
	
	public void reduce(Text key,Iterable<TextArrayWriteable> values,Context context) throws IOException,InterruptedException
	{
		float [] weights = null;
		boolean initArraySize=true;
		int recordCount=0;
		System.out.println("Inside reducer!");
		for(TextArrayWriteable recordArray:values)
		{
			Text record=new Text(recordArray.toStrings()[0]);
			System.out.println("record:"+record);
			String[] strWeights=record.toString().split(",");
			if (initArraySize)
			{
				weights=new float[strWeights.length-1];
				initArraySize=false;
			}
			for (int i=0;i<strWeights.length-1;i++)
			{
				
				weights[i]+=Float.parseFloat(strWeights[i])*Integer.parseInt(strWeights[strWeights.length-1]);
			}
			recordCount+=Integer.parseInt(strWeights[strWeights.length-1]);
		}
		
		String textWeight=new String();
		for (int i=0;i<weights.length;i++)
		{
			if (i==weights.length-1)
			{
				textWeight+=(weights[i]/recordCount);
			}
			else
			{
				textWeight+=weights[i]/recordCount+",";
			}
		}
		word.set(key+",");
		value.set(textWeight);
		context.write(word,value);
	}
}
