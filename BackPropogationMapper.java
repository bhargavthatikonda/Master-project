package proj9052507_BPN;

import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;

public class BackPropogationMapper extends Mapper<LongWritable,Text,Text,TextArrayWriteable> {

	private Text word = new Text();
	private TextArrayWriteable parts;
	
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		
		String[] values=value.toString().split(",");
		parts = new TextArrayWriteable(values);		
		word.set("1");
		context.write(word, parts);
	}
}
