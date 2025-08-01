#include <stdio.h>

class Task {
public:
    int id;
    Task *next;
    enum {
        TASK_TYPE_1,
        TASK_TYPE_2
    } type;
    // This struct is anonymous b/c it does not have a name
    // and it is not unnamed class.
    // Anonymous classes are a GNU extension.
    struct {
      int y;
    };
    // This struct is an unnamed class see [class.pre]p1
    // http://eel.is/c++draft/class#pre-1.sentence-6
    struct {
      int x;
    } my_type_is_nameless;
    struct name {
      int x;
      enum E : int {} e;
      enum E2 {} e2;
    } my_type_is_named;
    enum E : unsigned char {} e;
    union U {
    } u;
    static constexpr long static_constexpr_field = 47;
    static constexpr bool static_constexpr_bool_field = true;
    static int static_mutable_field;
    Task(int i, Task *n):
        id(i),
        next(n),
        type(TASK_TYPE_1)
    {}
};
int Task::static_mutable_field = 42;

template <unsigned Value> struct PointerInfo {
  enum Masks1 { pointer_mask };
  enum class Masks2 { pointer_mask };
};

template <unsigned Value, typename InfoType = PointerInfo<Value>>
struct Pointer {
  // When compiling for Windows with exceptions enabled, this struct
  // must contain something that takes space and is initialised.
  // Otherwise it will not be present in the debug information.
  int pad = 0;
};

enum EnumType {};
enum class ScopedEnumType {};
enum class EnumUChar : unsigned char {};

struct alignas(128) OverAlignedStruct {};
OverAlignedStruct over_aligned_struct;

struct WithNestedTypedef {
  typedef int TheTypedef;
};
WithNestedTypedef::TheTypedef typedefed_value;

int main (int argc, char const *argv[])
{
    Task *task_head = new Task(-1, NULL);
    Task *task1 = new Task(1, NULL);
    Task *task2 = new Task(2, NULL);
    Task *task3 = new Task(3, NULL); // Orphaned.
    Task *task4 = new Task(4, NULL);
    Task *task5 = new Task(5, NULL);

    task_head->next = task1;
    task1->next = task2;
    task2->next = task4;
    task4->next = task5;

    int total = 0;
    Task *t = task_head;
    while (t != NULL) {
        if (t->id >= 0)
            ++total;
        t = t->next;
    }
    printf("We have a total number of %d tasks\n", total);

    // This corresponds to an empty task list.
    Task *empty_task_head = new Task(-1, NULL);

    typedef int myint;
    myint myint_arr[] = {1, 2, 3};

    EnumType enum_type;
    ScopedEnumType scoped_enum_type;
    EnumUChar scoped_enum_type_uchar;

    Pointer<3> pointer;
    PointerInfo<3>::Masks1 mask1 = PointerInfo<3>::Masks1::pointer_mask;
    PointerInfo<3>::Masks2 mask2 = PointerInfo<3>::Masks2::pointer_mask;

    return 0; // Break at this line
}
