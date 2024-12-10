// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "TPU.v"
`include "global_buffer_bram.v"


module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output              rsp_valid,
  input               rsp_ready,
  output     [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

  // localparam [9:0] load_A = {7'd1, 3'd0};
  // localparam [9:0] load_B = {7'd2, 3'd0};
  // localparam [9:0] read_C0 = {7'd3, 3'd0};
  // localparam [9:0] run_it = {7'd4, 3'd0};
  // localparam [9:0] st_arg = {7'd5, 3'd0};

  // localparam [9:0] get_result = {7'd6, 3'd0};
  // localparam [9:0] read_C1 = {7'd7, 3'd0};
  // localparam [9:0] read_C2 = {7'd8, 3'd0};
  // localparam [9:0] read_C3 = {7'd9, 3'd0};

  // localparam [9:0] read_A = {7'd12, 3'd0};
  // localparam [9:0] read_B = {7'd13, 3'd0};


  // Trivial handshaking for a combinational CFU
  assign rsp_valid = cmd_valid;
  assign cmd_ready = rsp_ready;

  reg signed [31:0] InputOffset; //state
  reg signed [31:0] FilterOffset; //state
  // reg signed [31:0] acc; // state
  

  // SIMD multiply step:
  wire signed [15:0] prod_0, prod_1, prod_2, prod_3;
  reg  signed [15:0] prod_0_reg, prod_1_reg, prod_2_reg, prod_3_reg;
  assign prod_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + InputOffset)
                  * ($signed(cmd_payload_inputs_1[7 : 0] ) + FilterOffset) ;
  assign prod_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + InputOffset)
                  * ($signed(cmd_payload_inputs_1[15: 8]) + FilterOffset);
  assign prod_2 =  ($signed(cmd_payload_inputs_0[23:16]) + InputOffset)
                  * ($signed(cmd_payload_inputs_1[23:16]) + FilterOffset);
  assign prod_3 =  ($signed(cmd_payload_inputs_0[31:24]) + InputOffset)
                  * ($signed(cmd_payload_inputs_1[31:24]) + FilterOffset);

  wire signed [31:0] sum_prods;
  assign sum_prods = prod_0_reg + prod_1_reg + prod_2_reg + prod_3_reg;


  // wire calculating= (cal) 1:0;
  //
  // select output -- note that we're not fully decoding the 3 function_id bits
  //
  assign rsp_payload_outputs_0 = (cmd_valid && (cmd_payload_function_id == {7'd1, 3'd0})) ? A_data_out :
                                 (cmd_valid && (cmd_payload_function_id == {7'd2, 3'd0})) ? B_data_out :
                                 (cmd_valid && (cmd_payload_function_id == {7'd3, 3'd0})) ? C_data_out[31:0] :
                                 (cmd_valid && (cmd_payload_function_id == {7'd7, 3'd0})) ? C_data_out[63:32] :
                                 (cmd_valid && (cmd_payload_function_id == {7'd8, 3'd0})) ? C_data_out[95:64] :
                                 (cmd_valid && (cmd_payload_function_id == {7'd9, 3'd0})) ? C_data_out[127:96] :
                                 (cmd_valid && (cmd_payload_function_id == {7'd6, 3'd0})) ? busy :
                                 (cmd_valid && (cmd_payload_function_id == {7'd11, 3'd0}))? cals  :   
                                 (cmd_valid && (cmd_payload_function_id == {7'd12, 3'd0}))? srdhm_result :
                                 (cmd_valid && (cmd_payload_function_id == {7'd1, 3'd1})) ? 32'd0 :
                                 (cmd_valid && (cmd_payload_function_id == {7'd4, 3'd1})) ? sum_prods :
                                 (cmd_valid && (cmd_payload_function_id == {7'd14, 3'd0}))? calr  :   
                                 (cmd_valid && (cmd_payload_function_id == {7'd15, 3'd0}))? rdbpot_result :
                                                                                        2'd3 ;



  wire [11:0]    A_index        = (cmd_valid && (cmd_payload_function_id == {7'd1, 3'd0} )) ? cmd_payload_inputs_0[11:0]
                                                                                                           : A_index_TPU;        
  wire [31:0]    A_data_in      = (cmd_valid && (cmd_payload_function_id == {7'd1, 3'd0})) ? cmd_payload_inputs_1 : 32'd0;    
  wire           A_wr_en        = (cmd_valid && (cmd_payload_function_id == {7'd1, 3'd0})) ? 1 : 0;     
  wire [31:0]    A_data_out; 

  wire [11:0]     B_index       = (cmd_valid && (cmd_payload_function_id == {7'd2, 3'd0})) ? cmd_payload_inputs_0[11:0]
                                                                                                           : B_index_TPU;
  wire [31:0]     B_data_in     = (cmd_valid && (cmd_payload_function_id == {7'd2, 3'd0})) ? cmd_payload_inputs_1 : 32'd0;  
  wire            B_wr_en       = (cmd_valid && (cmd_payload_function_id == {7'd2, 3'd0})) ? 1 : 0;      
  wire [31:0]    B_data_out;  

  wire [11:0]     C_index       = (cmd_valid &&  ( (cmd_payload_function_id == {7'd3, 3'd0}) 
                                                || (cmd_payload_function_id == {7'd7, 3'd0})
                                                || (cmd_payload_function_id == {7'd8, 3'd0})
                                                || (cmd_payload_function_id == {7'd9, 3'd0}))) ? cmd_payload_inputs_0[11:0]
                                                                                     : C_index_TPU;      
  wire [127:0]   C_data_in      = C_data_in_TPU;  
  wire           C_wr_en        = C_wr_en_TPU;   
  wire [127:0]   C_data_out; 


  reg [7:0]      K;
  reg [7:0]      M;
  reg [7:0]      N;

  reg            in_valid;
  reg            calculating;
  reg [31:0]     offset;

  reg cals, calr;

  always @(posedge clk) begin
    if (cmd_valid && (cmd_payload_function_id == {7'd5, 3'd0})) begin
        K           <= cmd_payload_inputs_0[7:0];
        M           <= cmd_payload_inputs_1[7:0];
        N           <= cmd_payload_inputs_1[15:8];
    end
  end


  always @(posedge clk) begin
    if (cmd_valid && (cmd_payload_function_id == {7'd4, 3'd0})) begin
        in_valid       <= 1'b1;
        offset         <= cmd_payload_inputs_1;
    end
    else begin
        in_valid       <= 1'b0;
    end
  end

  // reg prod_val;

  always @(posedge clk)begin
    if(cmd_valid && (cmd_payload_function_id == {7'd0, 3'd1}))begin
      InputOffset <= cmd_payload_inputs_0;
      FilterOffset <= cmd_payload_inputs_1;
      // prod_val <= 1;
    end
    else if(cmd_valid && (cmd_payload_function_id == {7'd2, 3'd1}))begin
      prod_0_reg <= prod_0;
      prod_1_reg <= prod_1;
      prod_2_reg <= prod_2;
      prod_3_reg <= prod_3;
      // sum_prods <= prod_0 + prod_1 + prod_2 + prod_3;
      // prod_val <= 0;
    end
  end

  reg [31:0] srdhm_input_a;
  reg [31:0] srdhm_input_b;
  wire [31:0] srdhm_result;
  reg srdhm_input_valid;
  wire srdhm_output_valid;

  always @(posedge clk)begin
    if(cmd_valid && cmd_payload_function_id == {7'd10, 3'd0})begin
      cals <= 1;
      srdhm_input_a <= cmd_payload_inputs_0;
      srdhm_input_b <= cmd_payload_inputs_1;
      srdhm_input_valid <= 1;
    end
    else if(srdhm_input_valid)begin
      srdhm_input_valid <=0;
    end
    else if(srdhm_output_valid)begin
      cals <= 0;
    end
  end

  SRDHM srdhm (
        .clk(clk),
        .rst(reset),
        .a(srdhm_input_a),
        .b(srdhm_input_b),
        .input_valid(srdhm_input_valid),
        .srdhm_x(srdhm_result),
        .output_valid(srdhm_output_valid)
    );

  reg [31:0] rdbpot_input_a;
  reg [31:0] rdbpot_input_b;
  wire [31:0] rdbpot_result;
  reg rdbpot_input_valid;
  wire rdbpot_output_valid;

  always @(posedge clk)begin
    if(cmd_valid && cmd_payload_function_id == {7'd13, 3'd0})begin
      calr <= 1;
      rdbpot_input_a <= cmd_payload_inputs_0;
      rdbpot_input_b <= cmd_payload_inputs_1;
      rdbpot_input_valid <= 1;
    end
    else if(rdbpot_input_valid)begin
      rdbpot_input_valid <=0;
    end
    else if(rdbpot_output_valid)begin
      calr <= 0;
    end
  end

  RDBPOT rdbpot (
        .clk(clk),
        .rst(reset),
        .x(rdbpot_input_a),
        .exp(rdbpot_input_b),
        .input_valid(rdbpot_input_valid),
        .rdbpot_x(rdbpot_result),
        .output_valid(rdbpot_output_valid)
    );


  wire               busy;

  wire [31:0]        A_data_out_TPU = A_data_out;
  wire [31:0]        B_data_out_TPU = B_data_out;
  wire [127:0]       C_data_out_TPU = C_data_out;

//   TPU My_TPU(
//     .clk            (clk),     
//     .rst_n          (~reset),     
//     .in_valid       (in_valid),         
//     .K              (K), 
//     .M              (M), 
//     .N              (N), 
//     .busy           (busy),     
//     .A_wr_en        (),         
//     .A_index        (A_index_TPU),         
//     .A_data_in      (),         
//     .A_data_out     (A_data_out_TPU),         
//     .B_wr_en        (),         
//     .B_index        (B_index_TPU),         
//     .B_data_in      (),         
//     .B_data_out     (B_data_out_TPU),         
//     .C_wr_en        (C_wr_en_TPU),         
//     .C_index        (C_index_TPU),         
//     .C_data_in      (C_data_in_TPU),         
//     .C_data_out     (C_data_out_TPU)         
//   );

    wire [11:0]    A_index_TPU; 
    wire [11:0]    B_index_TPU; 
    wire [11:0]    C_index_TPU; 

    wire [127:0]    C_data_in_TPU; 

  TPU My_TPU(
    .clk            (clk),     
    .rst_n          (~reset),   
    .input_offset   (offset),     
    .in_valid       (in_valid),         
    .K              (K), 
    .M              (M), 
    .N              (N), 
    .busy           (busy),     
    .A_wr_en        (),         
    .A_index        (A_index_TPU),         
    .A_data_in      (),         
    .A_data_out     (A_data_out_TPU),         
    .B_wr_en        (),         
    .B_index        (B_index_TPU),         
    .B_data_in      (),         
    .B_data_out     (B_data_out_TPU),         
    .C_wr_en        (C_wr_en_TPU),         
    .C_index        (C_index_TPU),         
    .C_data_in      (C_data_in_TPU),         
    .C_data_out     (C_data_out_TPU)     
);


  global_buffer_bram #(
      .ADDR_BITS(12),
      .DATA_BITS(32)
  )
  gbuff_A(
      .clk(clk),
      .rst_n(reset),
      .ram_en(1'b1),
      .wr_en(A_wr_en),
      .index(A_index),
      .data_in(A_data_in),
      .data_out(A_data_out)
  );

  global_buffer_bram #(
      .ADDR_BITS(12),
      .DATA_BITS(32)
  ) gbuff_B(
      .clk(clk),
      .rst_n(reset),
      .ram_en(1'b1),
      .wr_en(B_wr_en),
      .index(B_index),
      .data_in(B_data_in),
      .data_out(B_data_out)
  );


  global_buffer_bram #(
      .ADDR_BITS(12),
      .DATA_BITS(128)
  ) gbuff_C(
      .clk(clk),
      .rst_n(reset),
      .ram_en(1'b1),
      .wr_en(C_wr_en),
      .index(C_index),
      .data_in(C_data_in),
      .data_out(C_data_out)
  );


endmodule

module SRDHM(
  input wire clk,
  input wire rst,
  input wire [31:0] a,
  input wire [31:0] b,
  input wire input_valid,
  output reg [31:0] srdhm_x,
  output reg output_valid
);

parameter IDLE = 3'd0; 
parameter CAL1 = 3'd1;   
parameter CAL2 = 3'd2;
parameter CAL3 = 3'd3;
parameter DONE = 3'd4;

reg [2:0] state;
reg [63:0] ab_64, ab_64_nudge;
reg [31:0] nudge;
wire  overflow = ((a == 32'h80000000) && (b == 32'h80000000));
wire [63:0] ab_64_nudge_neg = ~ab_64_nudge + 1'b1;

always @(posedge clk)begin
  if(rst)begin
    state <= IDLE;
    output_valid <= 0;
    srdhm_x <= 0;
  end
  else begin
    case(state)
        IDLE: begin
          output_valid =0;
          // srdhm_x <= 0;
          if(input_valid)begin
            state <= CAL1;
          end
        end
        CAL1: begin
          ab_64 <= $signed(a)*$signed(b);
          state <= CAL2;
        end
        CAL2: begin
          nudge <= (ab_64[63]) ? 64'hc0000001 : 64'h40000000;
          state <= CAL3;
        end
        CAL3: begin
          ab_64_nudge <= ab_64 + nudge;
          state <= DONE;
        end
        DONE: begin
          state <= IDLE;
          output_valid <= 1;
          srdhm_x <= (overflow) ? 32'h7fffffff : //ab_64_nudge[62:31];
                    (ab_64_nudge[63]) ? ~(ab_64_nudge_neg[62:31]) -1'b1 : ab_64_nudge[62:31]; //{ab_64_nudge[62:32], 1'b0}
        end
    endcase
  end
end

endmodule

module RDBPOT(
  input wire clk,
  input wire rst,
  input wire [31:0] x,
  input wire [31:0] exp,
  input wire input_valid,
  output reg [31:0] rdbpot_x,
  output reg output_valid
);

parameter IDLE = 3'd0; 
parameter CAL1 = 3'd1;   
parameter CAL2 = 3'd2;
parameter CAL3 = 3'd3;
parameter DONE = 3'd4;

reg [2:0] state;
// reg [63:0] ab_64, ab_64_nudge;
reg [31:0] mask, remainder, threshold;
// wire  overflow = ((a == 32'h80000000) && (b == 32'h80000000));
// wire [63:0] ab_64_nudge_neg = ~ab_64_nudge + 1'b1;

always @(posedge clk)begin
  if(rst)begin
    state <= IDLE;
    output_valid <= 0;
    rdbpot_x<= 0;
  end
  else begin
    case(state)
        IDLE: begin
          output_valid =0;
          // srdhm_x <= 0;
          if(input_valid)begin
            state <= CAL1;
          end
        end
        CAL1: begin
          mask <= (exp << 1) - 1'b1;
          state <= CAL2;
        end
        CAL2: begin
          remainder <= x & mask;
          threshold <= ($signed(mask) >>1) + x[31];
          state <= DONE;
        end
        DONE: begin
          state <= IDLE;
          output_valid <= 1;
          rdbpot_x <= ($signed(x) >> exp) + ($signed(remainder) > $signed(threshold));
        end
    endcase
  end
end

endmodule