"""
Test lldb-dap setBreakpoints request
"""

from dap_server import Source
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os


class TestDAP_breakpointEvents(lldbdap_testcase.DAPTestCaseBase):
    @skipUnlessDarwin
    def test_breakpoint_events(self):
        """
        This test sets a breakpoint in a shared library and runs and stops
        at the entry point of a program. When we stop at the entry point,
        the shared library won't be loaded yet. At this point the
        breakpoint should set itself, but not be verified because no
        locations are resolved. We will then continue and expect to get a
        breakpoint event that informs us that the breakpoint in the shared
        library is "changed" and the correct line number should be
        supplied. We also set a breakpoint using a LLDB command using the
        "preRunCommands" when launching our program. Any breakpoints set via
        the command interpreter should not be have breakpoint events sent
        back to VS Code as the UI isn't able to add new breakpoints to
        their UI. Code has been added that tags breakpoints set from VS Code
        DAP packets so we know the IDE knows about them. If VS Code is ever
        able to register breakpoints that aren't initially set in the GUI,
        then we will need to revise this.
        """
        main_source_basename = "main.cpp"
        main_source_path = os.path.join(os.getcwd(), main_source_basename)
        foo_source_basename = "foo.cpp"
        foo_source_path = os.path.join(os.getcwd(), foo_source_basename)
        main_bp_line = line_number("main.cpp", "main breakpoint 1")
        foo_bp1_line = line_number("foo.cpp", "foo breakpoint 1")
        foo_bp2_line = line_number("foo.cpp", "foo breakpoint 2")

        # Visual Studio Code Debug Adapters have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        program = self.getBuildArtifact("a.out")

        # Set a breakpoint after creating the target by running a command line
        # command. It will eventually resolve and cause a breakpoint changed
        # event to be sent to lldb-dap. We want to make sure we don't send a
        # breakpoint any breakpoints that were set from the command line.
        # Breakpoints that are set via the VS code DAP packets will be
        # registered and marked with a special keyword to ensure we deliver
        # breakpoint events for these breakpoints but not for ones that are not
        # set via the command interpreter.
        bp_command = "breakpoint set --file foo.cpp --line %u" % (foo_bp2_line)
        self.build_and_launch(program, preRunCommands=[bp_command])
        main_bp_id = 0
        foo_bp_id = 0
        # Set breakpoints and verify that they got set correctly
        dap_breakpoint_ids = []
        response = self.dap_server.request_setBreakpoints(
            Source(main_source_path), [main_bp_line]
        )
        self.assertTrue(response["success"])
        breakpoints = response["body"]["breakpoints"]
        for breakpoint in breakpoints:
            main_bp_id = breakpoint["id"]
            dap_breakpoint_ids.append("%i" % (main_bp_id))
            self.assertTrue(
                breakpoint["verified"], "expect main breakpoint to be verified"
            )

        response = self.dap_server.request_setBreakpoints(
            Source(foo_source_path), [foo_bp1_line]
        )
        self.assertTrue(response["success"])
        breakpoints = response["body"]["breakpoints"]
        for breakpoint in breakpoints:
            foo_bp_id = breakpoint["id"]
            dap_breakpoint_ids.append("%i" % (foo_bp_id))
            self.assertFalse(
                breakpoint["verified"], "expect foo breakpoint to not be verified"
            )

        # Flush the breakpoint events.
        self.dap_server.wait_for_breakpoint_events(timeout=5)

        # Continue to the breakpoint
        self.continue_to_breakpoints(dap_breakpoint_ids)

        verified_breakpoint_ids = []
        unverified_breakpoint_ids = []
        for breakpoint_event in self.dap_server.wait_for_breakpoint_events(timeout=5):
            breakpoint = breakpoint_event["body"]["breakpoint"]
            id = breakpoint["id"]
            if breakpoint["verified"]:
                verified_breakpoint_ids.append(id)
            else:
                unverified_breakpoint_ids.append(id)

        self.assertIn(main_bp_id, unverified_breakpoint_ids)
        self.assertIn(foo_bp_id, unverified_breakpoint_ids)

        self.assertIn(main_bp_id, verified_breakpoint_ids)
        self.assertIn(foo_bp_id, verified_breakpoint_ids)
