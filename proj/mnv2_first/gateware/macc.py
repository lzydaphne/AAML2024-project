#!/bin/env python
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nmigen import Cat, Signal, signed

from nmigen_cfu import all_words, Sequencer, SimpleElaboratable, tree_sum

from .post_process import PostProcessor
from .registerfile import Xetter


class Madd4Pipeline(SimpleElaboratable):
    """A 4-wide Multiply Add pipeline.

    Pipeline takes 2 additional cycles.

    f_data and i_data each contain 4 signed 8 bit values. The
    calculation performed is:

    result = sum((i_data[n] + offset) * f_data[n] for n in range(4))

    Public Interface
    ----------------
    offset: Signal(signed(8)) input
        Offset to be added to all inputs.
    f_data: Signal(32) input
        4 bytes of filter data to use next
    i_data: Signal(32) input
        4 bytes of input data data to use next
    result: Signal(signed(32)) output
        Result of the multiply and add
    """
    PIPELINE_CYCLES = 2

    def __init__(self):
        super().__init__()
        self.offset = Signal(signed(9))
        self.f_data = Signal(32)
        self.i_data = Signal(32)
        self.result = Signal(signed(32))

    def elab(self, m):
        # Product is 17 bits: 8 bits * 9 bits = 17 bits
        products = [Signal(signed(17), name=f"product_{n}") for n in range(4)]
        for i_val, f_val, product in zip(
                all_words(self.i_data, 8), all_words(self.f_data, 8), products):
            f_tmp = Signal(signed(9))
            m.d.sync += f_tmp.eq(f_val.as_signed())
            i_tmp = Signal(signed(9))
            m.d.sync += i_tmp.eq(i_val.as_signed() + self.offset)
            m.d.comb += product.eq(i_tmp * f_tmp)

        m.d.sync += self.result.eq(tree_sum(products))


class Accumulator(SimpleElaboratable):
    """An accumulator for a Madd4Pipline

    Public Interface
    ----------------
    add_en: Signal() input
        When to add the input
    in_value: Signal(signed(32)) input
        The input data to add
    clear: Signal() input
        Zero accumulator.
    result: Signal(signed(32)) output
        Result of the multiply and add
    """

    def __init__(self):
        super().__init__()
        self.add_en = Signal()
        self.in_value = Signal(signed(32))
        self.clear = Signal()
        self.result = Signal(signed(32))

    def elab(self, m):
        accumulator = Signal(signed(32))
        m.d.comb += self.result.eq(accumulator)
        with m.If(self.add_en):
            m.d.sync += accumulator.eq(accumulator + self.in_value)
            m.d.comb += self.result.eq(accumulator + self.in_value)
        # clear always resets accumulator next cycle, even if add_en is high
        with m.If(self.clear):
            m.d.sync += accumulator.eq(0)


class ByteToWordShifter(SimpleElaboratable):
    """Shifts bytes into a word.

    Bytes are shifted from high to low, so that result is little-endian, 
    with the "first" byte occupying the LSBs

    Public Interface
    ----------------
    shift_en: Signal() input
        When to shift the input
    in_value: Signal(8) input
        The input data to shift
    result: Signal(32) output
        Result of the shift
    """

    def __init__(self):
        super().__init__()
        self.shift_en = Signal()
        self.in_value = Signal(8)
        self.clear = Signal()
        self.result = Signal(32)

    def elab(self, m):
        register = Signal(32)
        m.d.comb += self.result.eq(register)

        with m.If(self.shift_en):
            calc = Signal(32)
            m.d.comb += [
                calc.eq(Cat(register[8:], self.in_value)),
                self.result.eq(calc),
            ]
            m.d.sync += register.eq(calc)


class Macc4Run4(Xetter):
    """Sequences a Madd4 and post processor to accumulate 4 input channel's worth of data.

    Parameters
    ----------------
    max_input_depth:
        Maximum number of input words (max input_depth)

    Public Interface
    ----------------
    input_depth: Signal(range(max_input_depth)) input
        Number of words in input

    madd4_start: Signal(1) output
        Notification that madd4 inputs have been read.
    madd4_inputs_ready: Signal() input
        Whether or not inputs for the madd4 are ready.

    acc_add_en: Signal() output
        Add to accumulator (when Madd is complete)

    pp_start: Signal(1) output
        Notification that PP inputs have been read

    shift_en: Signal(1) output
        Notification that PP output is ready for shifting out
    shift_result: Signal(32) input
        The post processed accumulator value - result of the post processor.
    """

    def __init__(self, max_input_depth):
        super().__init__()
        self.max_input_depth = max_input_depth
        self.input_depth = Signal(range(max_input_depth))
        self.madd4_start = Signal()
        self.madd4_inputs_ready = Signal()
        self.acc_add_en = Signal()
        self.pp_start = Signal()
        self.shift_en = Signal()
        self.shift_result = Signal(signed(32))

    def elab(self, m):
        # Sequence triggers every time a madd4 starts in order trigger accumulator
        m.submodules['madd_seq'] = madd4_seq = Sequencer(
            Madd4Pipeline.PIPELINE_CYCLES)
        # Sequence triggers on last madd4 for an output channel to start post processor
        m.submodules['last_madd_seq'] = last_madd4_seq = Sequencer(
            Madd4Pipeline.PIPELINE_CYCLES)
        # Seqence triggers when post processor starts - allows pp output to be captured
        m.submodules['pp_seq'] = pp_seq = Sequencer(
            PostProcessor.PIPELINE_CYCLES)

        # Number of output channels that have outstanding madds
        outstanding_madd4_channels = Signal(3)

        # Number of output channels that are complete
        complete_channels = Signal(3)

        def schedule_madd4():
            m.d.comb += [
                madd4_seq.inp.eq(1),
                self.madd4_start.eq(1),
            ]

        outstanding_madd4s = Signal(10)
        with m.If(self.start):
            m.d.sync += outstanding_madd4_channels.eq(4)
            m.d.sync += complete_channels.eq(0)
            # Start doing Madds
            with m.If(self.madd4_inputs_ready):
                schedule_madd4()
                m.d.sync += outstanding_madd4s.eq(self.input_depth - 1)
            with m.Else():
                m.d.sync += outstanding_madd4s.eq(self.input_depth)
        with m.Elif((outstanding_madd4_channels != 0) & self.madd4_inputs_ready):
            schedule_madd4()
            # Track outstanding and trigger "last" on last
            m.d.sync += outstanding_madd4s.eq(outstanding_madd4s - 1)
            with m.If(outstanding_madd4s == 1):
                m.d.comb += last_madd4_seq.inp.eq(1)
                m.d.sync += [
                    outstanding_madd4_channels.eq(outstanding_madd4_channels - 1),
                    outstanding_madd4s.eq(self.input_depth),
                ]

        # Tell accumulator to add as each madd is finished
        m.d.comb += self.acc_add_en.eq(madd4_seq.sequence[-1])

        # Tell Post processor to start on last Madd finish
        m.d.comb += self.pp_start.eq(last_madd4_seq.sequence[-1])
        m.d.comb += pp_seq.inp.eq(last_madd4_seq.sequence[-1])

        # On post processor finished, take result. When all channels done, declare finished
        with m.If(pp_seq.sequence[-1]):
            m.d.comb += self.shift_en.eq(1)
            m.d.sync += complete_channels.eq(complete_channels + 1)
            with m.If(complete_channels == 3):
                m.d.comb += self.done.eq(1)

        m.d.comb +=  self.output.eq(self.shift_result)