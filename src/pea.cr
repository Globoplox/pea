# Taken from here: https://github.com/dlidstrom/NeuralNetworkInAllLangs/blob/main/Go/neural.go
class Pea::Network

  def initialize(input input_count : UInt32, hidden hidden_count : UInt32, output output_count : UInt32, @rand : Random = Random::DEFAULT)
    # Network
    @weight_hidden = Array(Float64).new(input_count * hidden_count) { @rand.next_float - 0.5 }
    @bias_hidden = Array(Float64).new hidden_count, 0f64
    @weight_output = Array(Float64).new(output_count * hidden_count) { @rand.next_float - 0.5 }
    @bias_output = Array(Float64).new output_count, 0f64
    # Pre allocating for use during training
    @hidden = Array(Float64).new hidden_count, 0f64
    @output = Array(Float64).new output_count, 0f64
    @gradients_hidden = Array(Float64).new hidden_count, 0f64
    @gradients_output = Array(Float64).new output_count, 0f64
  end

  protected def sigmoid(n : Float64) : Float64
    1f64 / (1f64 + Math.exp(-n))
  end

  protected def sigmoid_prim(n : Float64) : Float64
    n * (1f64 - n)
  end

  protected def propagate(input, output, weight, bias)
    output.each_index do |output_index|
      sum = 0f64
      input.each_index do |input_index|
        sum += input[input_index] * weight[input_index * output.size + output_index]
      end
      output[output_index] = sigmoid(sum + bias[output_index])
    end
  end

  def predict(input : Array(Float64)) : Array(Float64)
    propagate input, @hidden, @weight_hidden, @bias_hidden
    propagate @hidden, @output, @weight_output, @bias_output
    return @output
  end

  def train(input : Array(Float64), expected : Array(Float64), learning_rate : Float64)
    predict input # mutate @output and @hidden

    # Backpropagation and gradient descent

    @output.each_index do |output_index|
      @gradients_output[output_index] = (@output[output_index] - expected[output_index]) * sigmoid_prim(@output[output_index])
    end

    @hidden.each_index do |hidden_index|
      sum = 0f64
      @output.each_index do |output_index|
        sum +=  @gradients_output[output_index] * @weight_output[hidden_index * @output.size + output_index]
      end
       @gradients_hidden[hidden_index] = sum * sigmoid_prim(@hidden[hidden_index])
    end

    @hidden.each_index do |hidden_index|
      @output.each_index do |output_index|
        @weight_output[hidden_index * @output.size + output_index] -= learning_rate * @gradients_output[output_index] * @hidden[hidden_index]
      end
    end

    input.each_index do |input_index|
      @hidden.each_index do |hidden_index|
        @weight_hidden[input_index * @hidden.size + hidden_index] -= learning_rate * @gradients_hidden[hidden_index] * input[input_index]
      end
    end

    @output.each_index do |output_index|
      @bias_output[output_index] -= learning_rate * @gradients_output[output_index]
    end

    @hidden.each_index do |hidden_index|
      @bias_hidden[hidden_index] -= learning_rate * @gradients_hidden[hidden_index]
    end
    
  end
end
