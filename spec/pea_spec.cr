require "./spec_helper"
require "http/client"
require "compress/zip"

class PredictableRNG
  include Random
  P = 2147483647u64
  A = 16807u64
  
  def initialize
    @current = 1
  end

  def next_u
    @current = (A * @current % P).to_i32
  end
  
  def next_float
    next_u.to_f64 / P
  end
end

def booly(u : Array(Float64)) : Array(Bool)
  u.map &.> 0.5
end

describe PredictableRNG do
  it "Generate predicted numbers" do
    rng = PredictableRNG.new

    expected = [
      7.82636925942561e-06,
      0.131537788143166,
      0.755604293083588,
      0.44134794289309,
      0.734872750814479,
      0.00631718803491313,
      0.172979253424788,
      0.262310957192588,
    ]

    numbers = Array(Float64).new(expected.size) { rng.next_float }

    puts "Generated numbers: #{numbers}"

    numbers.zip(expected).each do |got, expected|
      (got - expected).abs.should be < 0.0001
    end
  end
end

describe Pea do
  
  it "Learn logical functions" do

    dataset = Array({Array(Float64), Array(Float64)}).new
    0.upto 1 do |a|
      0.upto 1 do |b|
        dataset << {[a, b].map(&.to_f64), [a ^ b, 1 - (a ^ b), a | b, a & b, 1 - (a | b), 1 - (a & b)].map(&.to_f64)}
      end
    end
    
    network = Pea::Network.new input: 2, hidden: 2, output: 6, rand: PredictableRNG.new

    1000.times do
      dataset.each do |input, output|
        network.train input, output, learning_rate: 1f64
      end
    end

    dataset.each do |input, output|
	    got = network.predict input

      got = booly got
      output = booly output
      
      got.should eq output
    end
    
  end

  it "Learn to recognize digits" do

    data = nil
    if File.exists? "semeion.data"
      data = File.read_lines "semeion.data"
    else
      response = HTTP::Client.get "https://archive.ics.uci.edu/static/public/178/semeion+handwritten+digit.zip"
      raise "Could not fetch the semeion dataset: '#{response.body}'" unless response.success?
      Compress::Zip::Reader.open IO::Memory.new(response.body), &.each_entry do |entry|
        next unless entry.filename == "semeion.data"
        data = entry.io.gets_to_end.split "\r\n", remove_empty: true
      end
      raise "Could not find data file in the dataset" if data.nil?
      File.write "semeion.data", data.join '\n'
    end

    dataset = data.map do |line|
      entry = line.split ' '
      input = entry[start: 0, count: 16 * 16]
      output = entry[start: 256, count: 10]
      {input.map(&.to_f64), output.map(&.to_i.to_f64)}
    end

    dataset = dataset.shuffle

    # I get significantly higher accuracy when learning the whole dataset
    # (as shown in the csharp implem i think ?),
    # than when splitting into train/test
    # Im probably bullshitting away but wouldn't that mean that the network is overfit and just learned the train set as is ?
    # Learning the whole data set I can get over 99.5% accuracy
    # With a split 90/10 I get around 87% accuracy, varying from 83 to 90 between run.
    # With a 50/50 split I get around 84% accurcay, best 85.

    # I have played with hyper parameters (hidden layer size, learn/train, cycle and learning rate) and without overfit, i cant get past 85% accuracy.
    # What should I leverage ? Cutting into the weigth randomly during training to break patterns ?
    # Adding layers ?
    # Convolutions ? What are they ?

    train_set_size = 0.5
    # Split each subdataset, not the whole dataset
    # So we ensure it lean/test enough of each possible digit
    dataset = dataset.group_by(&.[1]).values.map do |dataset|
      train_set = dataset[...(dataset.size * train_set_size).floor.to_i]
      test_set = dataset[(dataset.size * train_set_size).floor.to_i..]
      {train_set, test_set}
    end
    train_set = dataset.flat_map &.[0]
    test_set = dataset.flat_map &.[1]
    
    network = Pea::Network.new input: 256, hidden: 14, output: 10

    20.times do
      train_set.shuffle.each do |input, output|
        network.train input, output, learning_rate: 0.5f64
      end
    end

    20.times do
      train_set.shuffle.each do |input, output|
        network.train input, output, learning_rate: 0.1f64
      end
    end
    
    failure = 0
    test_set.each do |input, output|
      got = network.predict input
      got = booly got
      output = booly output
      failure += 1 if got != output
    end

    accuracy = 100 - failure * 100 / test_set.size
    puts "Accuracy: #{accuracy}%"

    accuracy.should be > 95.0
  
 end
end
