#include <iostream>

#include "Auxiliary.h"

void exit_error(int line, const std::string &msg)
{
  std::cerr << "error(line : " << line << "): " << msg << std::endl;
  exit(1);
}

using vgdml::Auxiliary;

Auxiliary make_auxiliary(std::string t, std::string v, std::string u)
{
  Auxiliary a;
  a.SetType(t);
  a.SetValue(v);
  a.SetUnit(u);
  return a;
}

int main()
{
  // Basic construction/destruction
  {
    Auxiliary a;

    // No-throw, size is 0
    if (a.GetChildren().size() != 0) exit_error(__LINE__, "default construction failed");
  }

  // Copy/move construction/assignment
  {
    Auxiliary a = make_auxiliary("foo", "bar", "baz");
    a.AddChild(make_auxiliary("a", "b", "c"));
    a.AddChild(make_auxiliary("d", "e", "f"));

    // Copy
    {
      Auxiliary b{a};
      if (b != a) exit_error(__LINE__, "copy construction failed");

      Auxiliary c;
      c = a;
      if (c != a) exit_error(__LINE__, "copy assignment failed");
    }

    // Move
    {
      Auxiliary nullc;

      Auxiliary movedCons = a;
      Auxiliary b{std::move(movedCons)};

      if (b != a) exit_error(__LINE__, "move construction failed");
      if (movedCons != nullc) exit_error(__LINE__, "moved object in incorrect state");

      Auxiliary movedAssi = a;
      Auxiliary c;
      c = std::move(movedAssi);

      if (b != a) exit_error(__LINE__, "move assignment failed");
      if (movedAssi != nullc) exit_error(__LINE__, "moved object in incorrect state");
    }
  }

  return 0;
}