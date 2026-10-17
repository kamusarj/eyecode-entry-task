# Boot recovery in the recorded VM

Before the ROS experiments, the imported VM entered emergency mode during boot. Inspection found the EFI filesystem UUID and `/etc/fstab` entries were consistent. Kernel logs instead showed expedited-RCU stalls during device initialization.

A test boot with `rcupdate.rcu_normal=1` reached the desktop. The setting was then retained in the existing GRUB entries and this drop-in:

```sh
# /etc/default/grub.d/99-eyecode-part1-rcu.cfg
GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT rcupdate.rcu_normal=1"
```

The recorded VM subsequently booted and sustained the simulation/Nav2 workload. An early RCU warning remained; the underlying hypervisor/kernel mechanism was not independently established. This is a record of the fix used for that VM, not a prerequisite for a healthy Ubuntu/ROS installation.

Relevant logs are `evidence/debugging/boot_failure.log` and `evidence/debugging/boot_recovered.log`. No filesystem formatting or `/etc/fstab` change was required.

References: [Linux kernel parameter documentation](https://www.kernel.org/doc/html/v6.8/admin-guide/kernel-parameters.html) and the [similar VirtualBox report](https://github.com/VirtualBox/virtualbox/issues/861).
