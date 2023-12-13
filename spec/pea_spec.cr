require "./spec_helper"

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
end
