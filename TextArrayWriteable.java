package proj9052507_BPN;

import org.apache.hadoop.io.Text;

public class TextArrayWriteable extends ArrayWritable {
	
	public TextArrayWriteable() {
		super(Text.class);
		}

		public TextArrayWriteable(Text[] data) {
		super(Text.class, data);
		}

		public TextArrayWriteable(String[] data) {
		super(data);
		}
}
